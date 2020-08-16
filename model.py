import torch 
import torch as nn
from torchvision import transforms, models
import numpy as np

class PoseEstimator(object):
    def __init__(self, pretrained = True):
        self.pose_estimator = models.detection.keypointrcnn_resnet50_fpn(pretrained= pretrained)
        if torch.cuda.is_available():
            self.pose_estimator = self.pose_estimator.cuda()
        self.pose_estimator.eval()
    
    def __call__(self, frame):
        x = transforms.ToTensor()(frame)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            keypoints = self.pose_estimator([x])
        return {'keypoints': keypoints[0]}
    
    def get_keypoints(self, keypoints_dict, label = 1, score_threshold = 0.5):
        keypoints = []
        if keypoints_dict:
            for i in (keypoints_dict['labels'] == label).nonzero().view(-1):
                if keypoints_dict['scores'][i] > score_threshold:
                    keypoint = keypoints_dict['keypoints'][i].detach().cpu().squeeze().numpy()
                    keypoints.append(keypoint)
        return np.asarray(keypoints, dtype = np.int32)