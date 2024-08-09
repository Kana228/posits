# scripts/visualize.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_keypoints(keypoints_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for kp in keypoints_3d:
        ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2])
    plt.show()

if __name__ == "__main__":
    # Load 3D keypoints predictions
    keypoints_3d_predictions = np.load('outputs/keypoints_3d_predictions.npy')
    
    # Plot the 3D keypoints
    plot_3d_keypoints(keypoints_3d_predictions)
