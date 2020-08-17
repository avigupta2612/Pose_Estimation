import torch 
import torch as nn
from torchvision import transforms, models
import numpy as np

class PoseEstimator(object):
    def __init__(self):
        self.pose_estimator = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        if torch.cuda.is_available():
            self.pose_estimator = self.pose_estimator.cuda()
        self.pose_estimator.eval()
    
    def __call__(self, image):
        x = transforms.ToTensor()(image)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            keypoints = self.pose_estimator([x])
        return keypoints[0]
    
    @staticmethod
    def get_keypoints(dictionary, label=1, score_threshold=0.5):
        keypoints = []
        if dictionary:           
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    keypoint = dictionary['keypoints'][i].detach().cpu().squeeze().numpy()
                    keypoints.append(keypoint)
        return np.asarray(keypoints, dtype=np.int32)
    