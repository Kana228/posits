##3D Pose Estimation from Video using CNN and RNN

This repository contains code for 3D pose estimation from video data using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The project leverages MediaPipe for 2D keypoint extraction, TensorFlow/Keras for model building, and Hyperopt for hyperparameter optimization.

##Output of predicted 3d keypoints 

https://github.com/user-attachments/assets/dfebd7cc-f90c-428c-9e99-751e1f6c91a7



##Output of predicted movements


https://github.com/user-attachments/assets/12344956-e9bb-4713-8759-e73f53b19f77


Project Overview

##The goal of this project is to estimate 3D human poses from 2D video data. The pipeline includes:

2D Keypoint Extraction: Using MediaPipe to extract 2D keypoints from video frames.
CNN-based 3D Pose Prediction: A CNN model is trained to predict 3D keypoints from the 2D keypoints.
RNN-based Temporal Smoothing: An RNN model is used to precit future movements and smooth the 3D pose predictions across frames.
Hyperparameter Optimization: Hyperopt is used to find the optimal model parameters.
Combined Video Creation: A side-by-side video comparing the original footage and the 2D keypoints visualization.

