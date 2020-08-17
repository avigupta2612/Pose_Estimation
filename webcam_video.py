import cv2
import time
import numpy as np
from model import PoseEstimator
import matplotlib.pyplot as plt
from utils import *

def record_video(capture_time):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames = []
    while(int(time.time() - start_time) < capture_time):
        flag, frame = cap.read()
        if flag == 0:
            break
        frames.append(frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frames

def pose_estimator(frames):
    estimator = PoseEstimator()
    pose_frames = []
    for frame in frames:
        keypoints_dictionary = estimator(frame)
        keypoints = estimator.get_keypoints(keypoints_dictionary, score_threshold=0.99)
        pose_frames.append(keypoints)

    return pose_frames

def display_pose(frames, pose_frames):
    for frame, keypoints in zip(frames, pose_frames):
        frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_dst = cv2.merge([frame_dst] * 3)
        overlay_k = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        overlay_k = draw_keypoints(overlay_k, keypoints, radius=4, alpha=0.8)
        frame_dst = np.hstack((frame, overlay_k))
        frame_img = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_img)
        plt.show()
    cv2.destroyAllWindows()

frames = record_video(1)
print("Video Recorded")
print(np.shape(frames))
pose_frames = pose_estimator(frames)
print("Poses Recorded")
display_pose(frames, pose_frames)


