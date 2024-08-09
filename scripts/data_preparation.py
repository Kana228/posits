import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_frames(video_path):
    """Extract frames from a video file."""
    print(f"Starting to extract frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

def extract_keypoints_from_frame(frame):
    """Extract both 2D and 3D keypoints from a single video frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints_2d = []
        keypoints_3d = []
        for landmark in results.pose_landmarks.landmark:
            keypoints_2d.append((landmark.x, landmark.y))           # 2D coordinates
            keypoints_3d.append((landmark.x, landmark.y, landmark.z))  # 3D coordinates

        return np.array(keypoints_2d), np.array(keypoints_3d)
    else:
        return None, None

def extract_keypoints_from_video(video_path, save_dir):
    """Extract and save 2D and 3D keypoints from a video."""
    print(f"Starting to process video {video_path}...")
    frames = extract_frames(video_path)
    keypoints_2d_all_frames = []
    keypoints_3d_all_frames = []

    print(f"Extracting keypoints from frames...")
    for i, frame in enumerate(frames):
        keypoints_2d, keypoints_3d = extract_keypoints_from_frame(frame)
        if keypoints_2d is not None and keypoints_3d is not None:
            keypoints_2d_all_frames.append(keypoints_2d)
            keypoints_3d_all_frames.append(keypoints_3d)
        if (i + 1) % 100 == 0:  # Print progress every 100 frames
            print(f"Processed {i + 1} frames...")

    # Convert to numpy arrays and save them
    keypoints_2d_all_frames = np.array(keypoints_2d_all_frames)
    keypoints_3d_all_frames = np.array(keypoints_3d_all_frames)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving keypoints to {save_dir}...")
    np.save(os.path.join(save_dir, 'keypoints_2d.npy'), keypoints_2d_all_frames)
    np.save(os.path.join(save_dir, 'keypoints_3d.npy'), keypoints_3d_all_frames)

    print("Keypoints extraction and saving complete.")

if __name__ == "__main__":
    video_path = '/Users/kana/Desktop/computer vision/project/data/raw/dance.mp4'
    save_dir = '/Users/kana/Desktop/computer vision/project/data/processed/'

    extract_keypoints_from_video(video_path, save_dir)
