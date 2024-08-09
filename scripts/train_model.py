# scripts/train_model.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def build_model(input_shape, num_keypoints):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=2, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3 * num_keypoints),  # 3D coordinates for each keypoint
        tf.keras.layers.Reshape((num_keypoints, 3))  # Reshape to match the output shape of (num_keypoints, 3)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model



def load_data():
    X = np.load('/Users/kana/Desktop/computer vision/project/data/processed/keypoints_2d.npy')
    Y = np.load('/Users/kana/Desktop/computer vision/project/data/processed/keypoints_3d.npy')
    
    # No need to flatten X, just reshape to ensure it is 3D if necessary
    # X shape should be (num_samples, 33, 2)
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

def train_model(X_train, Y_train, input_shape):
    model = build_model(input_shape)
    
    print("Starting training...")
    for epoch in tqdm(range(50)):
        history = model.fit(X_train, Y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        print(f"Epoch {epoch + 1}/50, Loss: {history.history['loss'][-1]}, Validation Loss: {history.history['val_loss'][-1]}")
    
    return model

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Verify shapes
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    
    # Determine input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])  # Shape is now (33, 2)
    
    # Train the model
    model = train_model(X_train, Y_train, input_shape)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test loss: {test_loss}')
    
    # Save the model
    model.save('/Users/kana/Desktop/computer vision/project/models/pose_estimation_model.h5')
    print("Model saved successfully.")
