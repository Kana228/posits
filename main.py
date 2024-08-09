from scripts.data_preparation import extract_keypoints_from_video
from scripts.train_model import build_model
from scripts.inference import predict_3d_keypoints
from scripts.visualize import plot_3d_keypoints
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    # Define paths
    video_path = 'data/raw/dance.mp4'
    save_dir = 'data/processed/'
    model_path = 'models/pose_estimation_model.h5'
    outputs_dir = 'outputs/'

    # Step 1: Data Preparation
    print("Extracting keypoints from video...")
    extract_keypoints_from_video(video_path, save_dir)

    # Load 2D and 3D keypoints
    X = np.load(f'{save_dir}keypoints_2d.npy')
    Y = np.load(f'{save_dir}keypoints_3d.npy')

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Step 2: Model Training
    print("Training model...")
    num_keypoints = X_train.shape[1]  # Number of keypoints, as the first dimension is the number of keypoints
    input_shape = (X_train.shape[1], X_train.shape[2])  # Shape is now (num_keypoints, 2)
    model = build_model(input_shape, num_keypoints)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)


    # Evaluate and save the model
    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test loss: {test_loss}')
    model.save(model_path)

    # Step 3: Inference
    print("Predicting 3D keypoints...")
    keypoints_2d = np.load(f'{save_dir}keypoints_2d.npy')
    keypoints_3d_predictions = [predict_3d_keypoints(model, kp) for kp in keypoints_2d]

    # Save predictions
    np.save(f'{outputs_dir}keypoints_3d_predictions.npy', keypoints_3d_predictions)

    # Step 4: Visualization
    print("Visualizing 3D keypoints...")
    keypoints_3d_predictions = np.load(f'{outputs_dir}keypoints_3d_predictions.npy')
    plot_3d_keypoints(keypoints_3d_predictions)

if __name__ == "__main__":
    main()
    
