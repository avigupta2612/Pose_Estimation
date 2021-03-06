import cv2
from model import PoseEstimator
from utils import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description= 'Parse image path')
parser.add_argument('image_path', metavar='ipath', type=str,
                    help= 'Parse image path')
args = parser.parse_args()

img_path = args.image_path

# Read image from path
img = cv2.imread(img_path)

# Initialize model object
estimator = PoseEstimator()
keypoint_dict = estimator(img)
keypoints = estimator.get_keypoints(keypoint_dict, score_threshold= 0.9)

# Plot the generated Pose
frame_dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frame_dst = cv2.merge([frame_dst] * 3)
overlay_k = draw_body_connections(img, keypoints, thickness=2, alpha=0.7)
overlay_k = draw_keypoints(overlay_k, keypoints, radius=4, alpha=0.8)
frame_dst = np.hstack((img, overlay_k))
frame_img = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2RGB)
plt.imshow(frame_img)
plt.show()

