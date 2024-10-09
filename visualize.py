import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os
from keras.config import enable_unsafe_deserialization
from scipy.ndimage import gaussian_filter1d  # for smoothing predictions
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

enable_unsafe_deserialization()

# Constants
NUM_KEYPOINTS = 33
NUM_DIMENSIONS = 2
NUM_OUTPUT_DIMENSIONS = 3
TAIL_LENGTH = 10
SMOOTHING_SIGMA = 0.3  # Gaussian smoothing sigma
WINDOW_SIZE = 20
FUTURE_STEPS = 40

# Colors for different body parts
COLORS = {
    'left': (0, 255, 0),   # Green
    'right': (255, 0, 0),  # Blue
    'mid': (0, 255, 255),  # Yellow
}

# Keypoint history for tail effect
keypoint_history = {i: deque(maxlen=TAIL_LENGTH) for i in range(NUM_KEYPOINTS)}

# Mediapipe Pose Connections
mp_pose = mp.solutions.pose

POSE_CONNECTIONS = [
    # Upper body
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB),

    # Lower body
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),

    # Torso
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
    (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
]


def load_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video FPS: {fps}")
    
    keypoints_2d_list = []
    frames = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks and len(results.pose_landmarks.landmark) == NUM_KEYPOINTS:
                keypoints_2d = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
                keypoints_2d_list.append(keypoints_2d)
            else:
                # Handle frames where pose is not detected by repeating the last known keypoints
                if keypoints_2d_list:
                    keypoints_2d_list.append(keypoints_2d_list[-1])
                else:
                    # If no keypoints have been detected yet, append zeros
                    keypoints_2d_list.append([(0, 0) for _ in range(NUM_KEYPOINTS)])

    cap.release()
    return keypoints_2d_list, frames, fps

# CNN-based 3D pose prediction
def predict_3d_pose(cnn_model, keypoints_2d):
    keypoints_2d_array = np.array(keypoints_2d).reshape(1, NUM_KEYPOINTS, NUM_DIMENSIONS)
    keypoints_3d_pred = cnn_model.predict(keypoints_2d_array, verbose=0)  # Disable verbosity for faster execution
    return keypoints_3d_pred.reshape(NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)

#Handling Static Start and Gradual Movement**
def predict_movements(rnn_model, keypoints_3d_pred_list):
    # Pad the keypoints_3d_pred_list with the first frame to provide initial context
    if len(keypoints_3d_pred_list) < WINDOW_SIZE:
        padding = [keypoints_3d_pred_list[0]] * (WINDOW_SIZE - len(keypoints_3d_pred_list))
        keypoints_3d_pred_list = padding + keypoints_3d_pred_list

    predicted_movements = []

    # Loop through the keypoints_3d_pred_list with a sliding window
    for i in range(len(keypoints_3d_pred_list) - WINDOW_SIZE + 1):
        window = keypoints_3d_pred_list[i:i + WINDOW_SIZE]
        keypoints_3d_pred_array = np.array(window).reshape(1, WINDOW_SIZE, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)
        # Get the predicted movement for the current window
        movement = rnn_model.predict(keypoints_3d_pred_array)
        predicted_movements.append(movement)
    # Convert predicted movements to a numpy array
    predicted_movements = np.array(predicted_movements).reshape(-1, NUM_KEYPOINTS * NUM_OUTPUT_DIMENSIONS)

    # Apply Savitzky-Golay smoothing
    for i in range(NUM_KEYPOINTS):
        for j in range(NUM_OUTPUT_DIMENSIONS):
            window_length = 7 if len(predicted_movements) >= 7 else 3
            polyorder = 2 if window_length > 2 else 1
            predicted_movements[:, i * NUM_OUTPUT_DIMENSIONS + j] = savgol_filter(
                predicted_movements[:, i * NUM_OUTPUT_DIMENSIONS + j], window_length=window_length, polyorder=polyorder, mode='nearest'
            )
    # Apply pose constraints **after** smoothing
    predicted_movements = apply_pose_constraints(predicted_movements)
    return predicted_movements



# Interpolation functions for keypoints
def interpolate_keypoints(keypoints, num_frames):
    original_length = len(keypoints)
    original_indices = np.linspace(0, original_length - 1, num=original_length)
    new_indices = np.linspace(0, original_length - 1, num=num_frames)

    keypoints_interpolated = np.zeros((num_frames, NUM_KEYPOINTS, NUM_DIMENSIONS))
    for i in range(NUM_KEYPOINTS):
        x_interp = interp1d(original_indices, [kp[i][0] for kp in keypoints], kind='linear', fill_value="extrapolate")
        y_interp = interp1d(original_indices, [kp[i][1] for kp in keypoints], kind='linear', fill_value="extrapolate")
        keypoints_interpolated[:, i, 0] = x_interp(new_indices)
        keypoints_interpolated[:, i, 1] = y_interp(new_indices)

    return keypoints_interpolated

def interpolate_tracked_poses(poses, num_frames):
    if len(poses.shape) == 2:
        poses = poses.reshape(-1, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)

    original_length = poses.shape[0]
    original_indices = np.linspace(0, original_length - 1, num=original_length)
    new_indices = np.linspace(0, original_length - 1, num=num_frames)

    poses_interpolated = np.zeros((num_frames, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS))
    for i in range(NUM_KEYPOINTS):
        for j in range(NUM_OUTPUT_DIMENSIONS):
            keypoint_values = poses[:, i, j]  # Access specific dimension values for interpolation
            interp_func = interp1d(original_indices, keypoint_values, kind='linear', fill_value="extrapolate")
            poses_interpolated[:, i, j] = interp_func(new_indices)

    return poses_interpolated

def draw_keypoints_with_skeleton(shape, keypoints_2d, keypoints_3d, frame_index, is_future=False):
    frame = np.zeros(shape, dtype=np.uint8)
    keypoints_3d = keypoints_3d.reshape(NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
    keypoints_3d_projected = []
    for keypoint in keypoints_3d:
        x, y, z = keypoint

        scaling_factor = 500  # Adjusted scaling factor for better separation
        x_proj = int((x / (z + 1e-5)) * scaling_factor + shape[1] / 2)  
        y_proj = int((y / (z + 1e-5)) * scaling_factor + shape[0] / 2)  
        keypoints_3d_projected.append((x_proj, y_proj))
    if is_future:
        blend_factor = min(frame_index / FUTURE_STEPS, 1.0)
        color = (
            int((1 - blend_factor) * COLORS['mid'][0] + blend_factor * (0)),
            int((1 - blend_factor) * COLORS['mid'][1] + blend_factor * (0)),
            int((1 - blend_factor) * COLORS['mid'][2] + blend_factor * (255))
        )
    else:
        color = COLORS['mid']  # Yellow for tracking
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = (int(keypoints_2d[start_idx][0] * shape[1]), int(keypoints_2d[start_idx][1] * shape[0]))
        end_point = (int(keypoints_2d[end_idx][0] * shape[1]), int(keypoints_2d[end_idx][1] * shape[0]))
        cv2.line(frame, start_point, end_point, color, 2)

    # Draw the 3D projections (circles)
    for i, (x_proj, y_proj) in enumerate(keypoints_3d_projected):
        # Ensure keypoints are within frame boundaries
        if 0 <= x_proj < shape[1] and 0 <= y_proj < shape[0]:
            cv2.circle(frame, (x_proj, y_proj), 3, color, -1)

    return frame

def apply_pose_constraints(movements):
    for i in range(movements.shape[0]):  # Ensure you're within valid indices
        for j in range(NUM_KEYPOINTS):
            # Apply constraints to the z-dimension to avoid unrealistic poses
            if j * NUM_OUTPUT_DIMENSIONS + 2 < movements.shape[1]:
                movements[i, j * NUM_OUTPUT_DIMENSIONS + 2] = np.clip(
                    movements[i, j * NUM_OUTPUT_DIMENSIONS + 2], -1, 1
                )
    return movements


# Load pre-trained models
cnn_model = load_model('/Users/kana/Desktop/computer vision/project/models/cnnthai.keras')
rnn_model = load_model('/Users/kana/Desktop/computer vision/project/models/rnnthai.keras')

# Load video data and extract keypoints
video_path = '/Users/kana/Desktop/computer vision/project/data/raw/thai3.mp4'
keypoints_2d_list, frames, input_fps = load_video_data(video_path)


# Ensure that keypoints_2d_list has at least WINDOW_SIZE frames by padding if necessary
if len(keypoints_2d_list) < WINDOW_SIZE:
    padding = [keypoints_2d_list[0]] * (WINDOW_SIZE - len(keypoints_2d_list))
    keypoints_2d_list = padding + keypoints_2d_list

# Predict 3D poses using the CNN model
keypoints_3d_pred_list = [predict_3d_pose(cnn_model, keypoints_2d) for keypoints_2d in keypoints_2d_list]


# Predict movements using the RNN model with sliding window
predicted_movements = predict_movements(
    rnn_model, 
    keypoints_3d_pred_list[-min(WINDOW_SIZE, len(keypoints_3d_pred_list)):]  
)

# Define desired output video parameters based on input FPS
fps = input_fps if input_fps > 0 else 30.0  
desired_duration = 60  
total_frames = int(desired_duration * fps)
output_dir = '/Users/kana/Desktop/computer vision/project/output'
output_path = os.path.join(output_dir, 'output3d123.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (1280, 480)  

# Adjust the output movements shape to match total frames
if len(predicted_movements) < total_frames:
    # If predictions are fewer than total frames, pad with the last movement
    padding = total_frames - len(predicted_movements)
    last_movement = predicted_movements[-1] if predicted_movements.size > 0 else np.zeros((NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS))
    padding_frames = np.repeat(last_movement[np.newaxis, ...], padding, axis=0)
    predicted_movements = np.concatenate([predicted_movements, padding_frames], axis=0)


interpolated_movements = interpolate_tracked_poses(predicted_movements, WINDOW_SIZE)


print(f"Predicted movements shape: {predicted_movements.shape}")

# Apply pose constraints to the predicted movements
predicted_movements = apply_pose_constraints(predicted_movements)

# Interpolate the predicted movements to match the desired window size (if fewer than WINDOW_SIZE frames)
interpolated_movements = interpolate_tracked_poses(predicted_movements, WINDOW_SIZE)
print(f"Interpolated movements shape: {interpolated_movements.shape}")

#  reshape the array with the interpolated data
predicted_movements_reshaped = interpolated_movements.reshape(WINDOW_SIZE, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
print(f"Reshaped predicted movements shape: {predicted_movements_reshaped.shape}")


# Initialize the video writer once using the frame size (assuming it's 640x480 for the original video)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Interpolate keypoints and predicted movements to ensure they match the desired video length
keypoints_2d_list_interpolated = interpolate_keypoints(keypoints_2d_list, total_frames)
# Handle predicted_movements to match total_frames
num_predicted_movements = predicted_movements.shape[0]

if num_predicted_movements >= total_frames:
    predicted_movements_interpolated = predicted_movements[:total_frames]
else:
    # Pad the remaining frames with the last predicted movement
    padding = total_frames - num_predicted_movements
    if num_predicted_movements > 0:
        last_movement = predicted_movements[-1]
    else:
        last_movement = np.zeros((NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS))
    padding_frames = np.repeat(last_movement[np.newaxis, ...], padding, axis=0)
    predicted_movements_interpolated = np.concatenate([predicted_movements, padding_frames], axis=0)
# Apply Gaussian smoothing with tuned sigma

predicted_movements_interpolated = predicted_movements_interpolated.reshape(total_frames, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)

# Now apply Gaussian smoothing
for i in range(NUM_KEYPOINTS):
    for j in range(NUM_OUTPUT_DIMENSIONS):
        predicted_movements_interpolated[:, i, j] = gaussian_filter1d(
            predicted_movements_interpolated[:, i, j], sigma=SMOOTHING_SIGMA
        )
# Reshape the array with the interpolated data
predicted_movements_reshaped = predicted_movements_interpolated.reshape(total_frames, NUM_KEYPOINTS, NUM_OUTPUT_DIMENSIONS)
print(f"Final predicted movements reshaped shape: {predicted_movements_reshaped.shape}")



for i in range(total_frames):
    frame = cv2.resize(frames[i % len(frames)], (640, 480))  # Resize to half width for side-by-side
    current_movement = predicted_movements_reshaped[i]
    tracking_skeleton_frame = draw_keypoints_with_skeleton(
        (480, 640, 3),  
        keypoints_2d_list_interpolated[i], 
        keypoints_3d_pred_list[i % len(keypoints_3d_pred_list)],  
        i,
        is_future=True  
    )
    combined_skeleton_frame = tracking_skeleton_frame
    if i < total_frames - FUTURE_STEPS:
        for j in range(FUTURE_STEPS):
            if i + j < total_frames:  
                future_movement = predicted_movements_reshaped[i + j]
                future_skeleton_frame = draw_keypoints_with_skeleton(
                    (480, 640, 3), 
                    keypoints_2d_list_interpolated[i + j], 
                    future_movement,  
                    j,  
                    is_future=True
                )
                combined_skeleton_frame = cv2.addWeighted(combined_skeleton_frame, 0.6, future_skeleton_frame, 0.4, 0)

    combined_frame = np.concatenate((frame, combined_skeleton_frame), axis=1)
    video_writer.write(combined_frame)
    
video_writer.release()

print("Output video saved successfully.")
