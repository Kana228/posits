import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from hyperopt import fmin, tpe, Trials, hp


# constants
NUM_KEYPOINTS = 33
NUM_DIMENSIONS = 2
NUM_OUTPUT_DIMENSIONS = 3

#  Mediapipe Pose
mp_pose = mp.solutions.pose.Pose()

# Define a function to load video data
def load_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_2d_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image)
        if results.pose_landmarks:
            keypoints_2d = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
            keypoints_2d_list.append(keypoints_2d)
    cap.release()
    cv2.destroyAllWindows()
    return keypoints_2d_list

# create cnn model
def create_cnn_model(conv1d_filters, conv1d_kernel_size, max_pooling_pool_size, dense_units):
    model = Sequential([
        layers.Conv1D(conv1d_filters, conv1d_kernel_size, activation='relu', input_shape=(NUM_KEYPOINTS, NUM_DIMENSIONS)),
        layers.MaxPooling1D(max_pooling_pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS),
        layers.Reshape((NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# train cnn model
def train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_list, epochs):
    if not isinstance(cnn_model, tf.keras.Model):
        raise ValueError("Invalid cnn_model")
    if not isinstance(keypoints_2d_train_list, list) or len(keypoints_2d_train_list) < 1:
        raise ValueError("Invalid keypoints_2d_train_list")
    if not isinstance(keypoints_3d_list, list) or len(keypoints_3d_list) < 1:
        raise ValueError("Invalid keypoints")
    keypoints_2d_array = np.array([np.array(kp) for kp in keypoints_2d_list])
    keypoints_3d_array = np.array(keypoints_3d_list).reshape(-1, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
    min_samples = min(len(keypoints_2d_array), len(keypoints_3d_array))
    keypoints_2d_array = keypoints_2d_array[:min_samples]
    keypoints_3d_array = keypoints_3d_array[:min_samples]
    keypoints_2d_array = keypoints_2d_array.reshape(-1, NUM_KEYPOINTS, NUM_DIMENSIONS)
    
    class EarlyStopping(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs['loss'] < 0.01:
                self.model.stop_training = True
    
    # save cnn model during trainong
    early_stopping_callback = EarlyStopping()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/Users/kana/Desktop/computer vision/project/checkpoints/cnn_model_checkpoint.h5.keras', save_best_only=True, monitor='loss')
    
    cnn_model.fit(keypoints_2d_array, keypoints_3d_array, epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback])


# create an rn model
def create_rnn_model(lstm_units, dropout_rate):
    model = Sequential([
        layers.LSTM(lstm_units, input_shape=(None, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS), return_sequences=True),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dense(NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# train rnn 
def train_rnn_model(rnn_model, keypoints_3d_pred_list, epochs):
    sequence_length = len(keypoints_3d_pred_list)
    keypoints_3d_pred_array = np.array(keypoints_3d_pred_list).reshape(-1, sequence_length, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    
    class EarlyStopping(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs['loss'] < 0.01:
                self.model.stop_training = True
    
    #save rnn model during training also
    early_stopping_callback = EarlyStopping()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/Users/kana/Desktop/computer vision/project/checkpoints/rnn_model_checkpoint.h5.keras', save_best_only=True, monitor='loss')
    
    rnn_model.fit(keypoints_3d_pred_array, keypoints_3d_pred_array, epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback])


# function to evaluate the models
def evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list):
    keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_val_list]
    tracked_poses = track_poses(rnn_model, keypoints_3d_pred_list)
    tracked_poses = tracked_poses.reshape(-1, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
    val_loss = np.mean((tracked_poses - keypoints_3d_val_list) ** 2)
    return val_loss

def predict_3d_pose(cnn_model, keypoints_2d):
    if not isinstance(keypoints_2d, list) or len(keypoints_2d) != NUM_KEYPOINTS:
        raise ValueError("Invalid input keypoints_2d")
    keypoints_2d_array = np.array(keypoints_2d).reshape(1, NUM_KEYPOINTS, NUM_DIMENSIONS)
    keypoints_3d_pred = cnn_model.predict(keypoints_2d_array)
    return keypoints_3d_pred[0]

def track_poses(rnn_model, keypoints_3d_pred_list):
    if not isinstance(keypoints_3d_pred_list, list) or len(keypoints_3d_pred_list) < 1:
        raise ValueError("Invalid input keypoints_3d_pred_list")
    keypoints_3d_pred_array = np.array(keypoints_3d_pred_list).reshape(1, len(keypoints_3d_pred_list), NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    tracked_poses = rnn_model.predict(keypoints_3d_pred_array)
    return tracked_poses[0]


# Load video data
video_path = '/Users/kana/Desktop/computer vision/project/data/raw/dance.mp4'
keypoints_2d_list = load_video_data(video_path)

# Split data into training and validation sets
num_train_samples = int(0.8 * len(keypoints_2d_list))
keypoints_2d_train_list = keypoints_2d_list[:num_train_samples]
keypoints_2d_val_list = keypoints_2d_list[num_train_samples:]

# Generate 3D keypoints for training and validation sets
keypoints_3d_list = [np.random.rand(NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS) for _ in keypoints_2d_train_list]
keypoints_3d_val_list = [np.random.rand(NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS) for _ in keypoints_2d_val_list]

# function for hyperparameter optimization
def objective(params):
    cnn_model = create_cnn_model(
        conv1d_filters=int(params['cnn_conv1d_filters']),
        conv1d_kernel_size=int(params['cnn_conv1d_kernel_size']),
        max_pooling_pool_size=int(params['cnn_max_pooling_pool_size']),
        dense_units=int(params['cnn_dense_units'])
    )
    train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_list, epochs=int(params['epochs']))

    rnn_model = create_rnn_model(
        lstm_units=int(params['rnn_lstm_units']),
        dropout_rate=params['rnn_dropout_rate']
    )
    keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_train_list]
    train_rnn_model(rnn_model, keypoints_3d_pred_list, epochs=int(params['epochs']))

    val_loss = evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list)
    return val_loss

# hyperparameter search space params
space = {
    'cnn_conv1d_filters': hp.quniform('cnn_conv1d_filters', 16, 128, 16),
    'cnn_conv1d_kernel_size': hp.quniform('cnn_conv1d_kernel_size', 2, 6, 1),
    'cnn_max_pooling_pool_size': hp.quniform('cnn_max_pooling_pool_size', 2, 5, 1),
    'cnn_dense_units': hp.quniform('cnn_dense_units', 64, 256, 32),
    'rnn_lstm_units': hp.quniform('rnn_lstm_units', 64, 256, 32),
    'rnn_dropout_rate': hp.uniform('rnn_dropout_rate', 0.1, 0.5),
    'epochs': hp.quniform('epochs', 10, 50, 10)
}

# Perform hyperparameter search with Hyperopt
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=30, trials=trials)



# train  best model
cnn_model = create_cnn_model(
    conv1d_filters=int(best['cnn_conv1d_filters']),
    conv1d_kernel_size=int(best['cnn_conv1d_kernel_size']),
    max_pooling_pool_size=int(best['cnn_max_pooling_pool_size']),
    dense_units=int(best['cnn_dense_units'])
)

train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_list, epochs=int(best['epochs']))

rnn_model = create_rnn_model(
    lstm_units=int(best['rnn_lstm_units']),
    dropout_rate=best['rnn_dropout_rate']
)
keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_train_list]
train_rnn_model(rnn_model, keypoints_3d_pred_list, epochs=int(best['epochs']))


# Evaluate the best model on the validation set
val_loss = evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list)
print(f'Best model validation loss: {val_loss:.4f}')

def draw_keypoints_on_black(shape, keypoints_2d):
    frame = np.zeros(shape, dtype=np.uint8)
    for (x, y) in keypoints_2d:
        x = int(x * shape[1])
        y = int(y * shape[0])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame


def create_combined_video(video_path, keypoints_2d_list, output_path):
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    total_frames = len(keypoints_2d_list)
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_index}")
            break
        
        if frame_index % 100 == 0:
            print(f"Creating video: frame {frame_index}/{total_frames}")
        
        keypoints_2d = keypoints_2d_list[frame_index]
        black_frame_with_keypoints = draw_keypoints_on_black(frame.shape, keypoints_2d)
        
        combined_frame = np.hstack((frame, black_frame_with_keypoints))
        out.write(combined_frame)
    
    cap.release()
    out.release()
    print(f"Video saved as {output_path}")

# Call the function to create a combined video
video_path = '/Users/kana/Desktop/computer vision/project/data/raw/dance.mp4'
keypoints_2d_list = load_video_data(video_path)
output_path = '/Users/kana/Desktop/computer vision/project/output/combined_video.mp4'
create_combined_video(video_path, keypoints_2d_list, output_path)
