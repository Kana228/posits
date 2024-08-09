# scripts/inference.py

import numpy as np
import tensorflow as tf

def predict_3d_keypoints(model, keypoints_2d):
    # Reshape keypoints_2d to match the input shape the model expects
    keypoints_2d = keypoints_2d.reshape(1, keypoints_2d.shape[0], keypoints_2d.shape[1])  # Shape: (1, 33, 2)
    
    # Make predictions
    keypoints_3d = model.predict(keypoints_2d)
    
    # Remove the batch dimension for easier handling
    keypoints_3d = keypoints_3d.reshape(keypoints_3d.shape[1], keypoints_3d.shape[2])  # Shape: (33, 3)
    
    return keypoints_3d


if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model('models/pose_estimation_model.h5')
    
    # Load the 2D keypoints
    keypoints_2d = np.load('data/keypoints_2d.npy')

    # Predict 3D keypoints
    keypoints_3d_predictions = [predict_3d_keypoints(model, kp) for kp in keypoints_2d]
    
    # Save the predictions
    np.save('outputs/keypoints_3d_predictions.npy', keypoints_3d_predictions)
