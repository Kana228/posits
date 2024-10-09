import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from hyperopt import fmin, tpe, Trials, hp
import matplotlib.pyplot as plt
from tensorflow import keras

# Constants
NUM_KEYPOINTS = 33
NUM_DIMENSIONS = 2
NUM_OUTPUT_DIMENSIONS = 3
fsteps = 30

# Mediapipe Pose
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

def normalize_keypoints(keypoints, frame_width=1280, frame_height=720):
    """ Normalize keypoints to the range [0, 1] using fixed frame dimensions. """
    return [(kp[0] / frame_width, kp[1] / frame_height) for kp in keypoints]

def augment_keypoints(keypoints, scale=0.05, noise=0.02):
    """ Reduce augmentation scale and noise to avoid instability. """
    keypoints_aug = [(x * (1 + scale * np.random.uniform(-1, 1)),
                      y * (1 + scale * np.random.uniform(-1, 1)))
                     for (x, y) in keypoints]
    return [(x + noise * np.random.randn(), y + noise * np.random.randn()) for (x, y) in keypoints_aug]



from tensorflow.keras.layers import Add

def create_cnn_model(conv1d_filters=64, conv1d_kernel_size=3, max_pooling_pool_size=2, dense_units=128):
    input_shape = (NUM_KEYPOINTS, NUM_DIMENSIONS)
    
    model_input = layers.Input(shape=input_shape)
    x = layers.Conv1D(conv1d_filters, conv1d_kernel_size, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    
    # Residual Block
    residual = layers.Conv1D(conv1d_filters, 1, padding='same')(x)  
    x = layers.Conv1D(conv1d_filters, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(max_pooling_pool_size)(x)
    residual = layers.MaxPooling1D(max_pooling_pool_size)(residual)  
    x = Add()([x, residual])
    x = layers.Dropout(0.3)(x)
    
    # Additional Layers
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)  
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    model_output = layers.Dense(NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)(x)
    model_output = layers.Reshape((NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS))(model_output)
    
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_list, epochs, batch_size=32):
    keypoints_2d_array = np.array([np.array(kp) for kp in keypoints_2d_train_list])
    keypoints_3d_array = np.array(keypoints_3d_list).reshape(-1, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
    min_samples = min(len(keypoints_2d_array), len(keypoints_3d_array))
    keypoints_2d_array = keypoints_2d_array[:min_samples]
    keypoints_3d_array = keypoints_3d_array[:min_samples]

    keypoints_2d_array = keypoints_2d_array.reshape(-1, NUM_KEYPOINTS, NUM_DIMENSIONS)
    
    indices = np.arange(keypoints_2d_array.shape[0])
    np.random.shuffle(indices)
    keypoints_2d_array = keypoints_2d_array[indices]
    keypoints_3d_array = keypoints_3d_array[indices]

    # Callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/Users/kana/Desktop/computer vision/project/checkpoints/cnn_model_checkpoint.h5.keras', save_best_only=True, monitor='val_loss')
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Training
    cnn_model.fit(
        keypoints_2d_array, keypoints_3d_array,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]
    )
def create_rnn_model(lstm_units, dropout_rate, future_steps):
    model = Sequential([
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, activation='relu'), 
                            input_shape=(20, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, activation='relu')),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, activation='relu')),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.RepeatVector(future_steps),  # Repeat the context vector for future steps
        layers.TimeDistributed(layers.Dense(NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS, activation='linear')),
        layers.Lambda(lambda x: x[:, -future_steps:, :])  # Ensure output shape matches future_steps
    ])
    model.compile(optimizer='adam', loss='mse')
    return model




def train_rnn_model(rnn_model, keypoints_3d_train_list, future_steps, epochs, batch_size=32):
    sequence_length = 20

    x_train = np.array([keypoints_3d_train_list[i:i + sequence_length] 
                        for i in range(len(keypoints_3d_train_list) - sequence_length - future_steps)])
    y_train = np.array([keypoints_3d_train_list[i + sequence_length: i + sequence_length + future_steps] 
                        for i in range(len(keypoints_3d_train_list) - sequence_length - future_steps)])

    x_train = x_train.reshape(-1, sequence_length, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    y_train = y_train.reshape(-1, future_steps, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)

    x_train = x_train.squeeze()

    # Callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/Users/kana/Desktop/computer vision/project/checkpoints/rnn_model_checkpoint.h5.keras', save_best_only=True, monitor='val_loss')
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Training
    rnn_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1
    )
    
    return rnn_model
def predict_future_poses(rnn_model, keypoints_3d_pred_list, future_steps):
    keypoints_3d_pred_array = np.array(keypoints_3d_pred_list).reshape(1, len(keypoints_3d_pred_list), NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    future_keypoints_3d_pred = rnn_model.predict(keypoints_3d_pred_array)
    return future_keypoints_3d_pred[0]

def evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list):
    keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_val_list]
    tracked_poses = track_poses(rnn_model, keypoints_3d_pred_list)
    
    tracked_poses = tracked_poses.reshape(-1, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
    min_samples = min(len(tracked_poses), len(keypoints_3d_val_list))
    tracked_poses = tracked_poses[:min_samples]
    keypoints_3d_val_list = keypoints_3d_val_list[:min_samples]

    val_loss = np.mean((tracked_poses - keypoints_3d_val_list) ** 2)
    return val_loss

def predict_3d_pose(cnn_model, keypoints_2d):
    keypoints_2d_array = np.array(keypoints_2d).reshape(1, NUM_KEYPOINTS, NUM_DIMENSIONS)
    keypoints_3d_pred = cnn_model.predict(keypoints_2d_array)
    return keypoints_3d_pred[0]

def track_poses(rnn_model, keypoints_3d_pred_list):
    keypoints_3d_pred_array = np.array(keypoints_3d_pred_list).reshape(1, len(keypoints_3d_pred_list), NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
    tracked_poses = rnn_model.predict(keypoints_3d_pred_array)
    return tracked_poses[0]

# Load video data
video_path = '/Users/kana/Desktop/computer vision/project/data/raw/thai2.mp4'
keypoints_2d_list = load_video_data(video_path)

# Split data into training and validation sets
num_train_samples = int(0.8 * len(keypoints_2d_list))
keypoints_2d_train_list = keypoints_2d_list[:num_train_samples]
keypoints_2d_val_list = keypoints_2d_list[num_train_samples:]

# Predict 3D keypoints using the CNN model
cnn_model = create_cnn_model(
    conv1d_filters=64,
    conv1d_kernel_size=3,
    max_pooling_pool_size=2,
    dense_units=128
)
keypoints_3d_train_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_train_list]
keypoints_3d_val_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_val_list]

# Hyperparameter optimization objective function
def objective(params):
    cnn_model = create_cnn_model(
        conv1d_filters=int(params['cnn_conv1d_filters']),
        conv1d_kernel_size=int(params['cnn_conv1d_kernel_size']),
        max_pooling_pool_size=int(params['cnn_max_pooling_pool_size']),
        dense_units=int(params['cnn_dense_units'])
    )
    train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_train_list, epochs=int(params['epochs']))

    rnn_model = create_rnn_model(
        lstm_units=int(params['rnn_lstm_units']),
        dropout_rate=params['rnn_dropout_rate'],
        future_steps=fsteps
    )

    keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_train_list]
    train_rnn_model(rnn_model, keypoints_3d_pred_list, future_steps=fsteps, epochs=int(params['epochs']))

    val_loss = evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list)
    return val_loss

# Hyperparameter search space
space = {
    'cnn_conv1d_filters': hp.quniform('cnn_conv1d_filters', 16, 128, 16),
    'cnn_conv1d_kernel_size': hp.quniform('cnn_conv1d_kernel_size', 2, 6, 1),
    'cnn_max_pooling_pool_size': hp.quniform('cnn_max_pooling_pool_size', 2, 5, 1),
    'cnn_dense_units': hp.quniform('cnn_dense_units', 64, 256, 32),
    'rnn_lstm_units': hp.quniform('rnn_lstm_units', 64, 256, 32),
    'rnn_dropout_rate': hp.uniform('rnn_dropout_rate', 0.1, 0.5),
    'epochs': hp.quniform('epochs', 5, 15, 5)
}

# Perform hyperparameter search with Hyperopt
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

# Train the best models
cnn_model = create_cnn_model(
    conv1d_filters=int(best['cnn_conv1d_filters']),
    conv1d_kernel_size=int(best['cnn_conv1d_kernel_size']),
    max_pooling_pool_size=int(best['cnn_max_pooling_pool_size']),
    dense_units=int(best['cnn_dense_units'])
)
train_cnn_model(cnn_model, keypoints_2d_train_list, keypoints_3d_train_list, epochs=int(best['epochs']))

rnn_model = create_rnn_model(
    lstm_units=int(best['rnn_lstm_units']),
    dropout_rate=best['rnn_dropout_rate'],
    future_steps=fsteps
)
keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_train_list]
train_rnn_model(rnn_model, keypoints_3d_pred_list, future_steps=fsteps, epochs=int(best['epochs']))

def plot_keypoints_2d(keypoints):
    plt.scatter([x for (x, y) in keypoints], [y for (x, y) in keypoints])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.show()

print('end of 2d')
for i in range(5):
    plot_keypoints_2d(keypoints_2d_list[i])
    plot_keypoints_2d(normalize_keypoints(keypoints_2d_list[i]))

def plot_3d_keypoints(keypoints_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = keypoints_3d[:, 0]
    ys = keypoints_3d[:, 1]
    zs = keypoints_3d[:, 2]
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Final evaluation of the models
val_loss = evaluate_models(cnn_model, rnn_model, keypoints_2d_val_list, keypoints_3d_val_list)
print(f"Validation Loss: {val_loss}")

num_samples_to_plot = 10
for i in range(num_samples_to_plot):
    keypoints_3d_pred = predict_3d_pose(cnn_model, keypoints_2d_val_list[i])
    plot_3d_keypoints(keypoints_3d_pred)

# Save the trained models
cnn_model.save('/Users/kana/Desktop/computer vision/project/models/cnnthai.keras', include_optimizer=False)
rnn_model.save('/Users/kana/Desktop/computer vision/project/models/rnnthai.keras', include_optimizer=False)
