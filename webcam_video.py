import cv2
import time
from model import PoseEstimator

capture_duration = 10
estimator = PoseEstimator()
pose_frames = []
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
start_time = time.time()
while(int(time.time() - start_time) < capture_duration):
    flag, frame = cap.read()
    if flag == 0:
        break
    out.write(frame)
    #keypoints_dictionary = estimator(frame)
    #keypoints = estimator.get_keypoints(keypoints_dictionary['keypoints'], score_threshold = 0.99)
    #pose_frames.append(keypoints)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print(pose_frames)