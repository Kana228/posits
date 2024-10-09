##3D Pose Estimation from Video using CNN and RNN

This repository contains code for 3D pose estimation from video data using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The project leverages MediaPipe for 2D keypoint extraction, TensorFlow/Keras for model building, and Hyperopt for hyperparameter optimization.

##Table of Contents

Project Overview
Installation
Usage
Project Structure
Model Architecture
Hyperparameter Optimization
Video Processing
Evaluation
Acknowledgements

https://github.com/user-attachments/assets/12344956-e9bb-4713-8759-e73f53b19f77


Project Overview

##The goal of this project is to estimate 3D human poses from 2D video data. The pipeline includes:

2D Keypoint Extraction: Using MediaPipe to extract 2D keypoints from video frames.
CNN-based 3D Pose Prediction: A CNN model is trained to predict 3D keypoints from the 2D keypoints.
RNN-based Temporal Smoothing: An RNN model is used to track and smooth the 3D pose predictions across frames.
Hyperparameter Optimization: Hyperopt is used to find the optimal model parameters.
Combined Video Creation: A side-by-side video comparing the original footage and the 2D keypoints visualization.
Installation

##To run the code, follow these steps:

Clone the repository:
bash
Copy code
git clone https://github.com/Kana228/posits.git
cd 3d-pose-estimation
Install the required Python packages:
bash
Copy code
pip install -r requirements.txt
Ensure you have the necessary video file in the correct path:
Place your video file in data/raw/dance.mp4.
Usage

##Training the Model
Run Hyperparameter Optimization:
python
Copy code
python main.py
Train the Best Model: The best CNN and RNN models from the optimization process will be retrained with the optimal parameters.
Evaluate the Model: The validation loss will be printed after evaluating the trained model on the validation set.
Create a Combined Video
To create a video that shows the original footage alongside the 2D keypoint visualization:

##python
Copy code
python create_video.py
Example Command:
python
Copy code
python create_video.py --video_path data/raw/dance.mp4 --output_path output/combined_video.mp4
Project Structure

##main.py: Contains the main training loop and hyperparameter optimization logic.
create_video.py: Contains the logic for creating a combined video of original and 2D keypoint visualizations.
data/: Directory to store raw video data.
output/: Directory to store output videos and model checkpoints.
requirements.txt: Python dependencies.
Model Architecture

##CNN Model
The CNN model is designed to predict 3D keypoints from 2D keypoints. It consists of:

##Conv1D layers for feature extraction.
MaxPooling layers for downsampling.
Dense layers for final pose prediction.
RNN Model
The RNN model is designed to track and smooth 3D keypoints across frames. It consists of:

##LSTM layers to capture temporal dependencies.
Dense layers for the final smoothed pose prediction.
Hyperparameter Optimization

##Hyperopt is used to optimize the following parameters:

Number of Conv1D filters
Conv1D kernel size
Max pooling size
Dense units in the CNN
LSTM units in the RNN
Dropout rate in the RNN
Number of training epochs
Video Processing

##The code includes utilities to:

Load video data and extract 2D keypoints using MediaPipe.
Create combined videos showing both the original frames and keypoints.
Evaluation

The evaluation function computes the validation loss based on the difference between predicted and actual 3D keypoints.

##Acknowledgements

MediaPipe for 2D pose estimation.
TensorFlow for model building.
Hyperopt for hyperparameter optimization.
