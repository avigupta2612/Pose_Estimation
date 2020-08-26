import cv2
import numpy as np
from model import PoseEstimator
from utils import *
import argparse

parser = argparse.ArgumentParser(description= 'Parse Video path')
parser.add_argument('video_path', metavar= 'path', type=str,
                    help= 'Pass the video path')
args = parser.parse_args()
print(args.video_path)

estimator = PoseEstimator()
videoclip = cv2.VideoCapture(args.video_path)

while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    
    pred_dict = estimator(frame)
    keypoints = estimator.get_keypoints(pred_dict, score_threshold=0.9)
    
    frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_dst = cv2.merge([frame_dst] * 3)
    overlay_k = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    overlay_k = draw_keypoints(overlay_k, keypoints, radius=4, alpha=0.8)
    frame_dst = np.hstack((frame, overlay_k))
    
    cv2.imshow('Video Demo', frame_dst)
    if cv2.waitKey(20) & 0xff == 27: 
        break

videoclip.release()
cv2.destroyAllWindows()
