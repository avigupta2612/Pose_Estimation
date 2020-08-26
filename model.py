import torch 
import torch as nn
from torchvision import transforms, models
import numpy as np

class PoseEstimator(object):
    """
    Pose Estimator Model Class

    Model Input : Image- Tensor of shape [C, H, W]

    Model Output : Dictionary Containing{
                    Boxes- Float Tensor of shape [N,4]
                    Labels- Int Tensor of shape [N]
                    Keypoints- Float Tensor of shape [N, K, 3]
                    Keypoint_scores- Float Tensor of shape [K]}
    """
    def __init__(self):
        self.pose_estimator = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        if torch.cuda.is_available():
            self.pose_estimator = self.pose_estimator.cuda()
        
        # Set model in Evaluation mode
        self.pose_estimator.eval()
    
    def __call__(self, image):
        # Transform the image to pytorch Tensor
        x = transforms.ToTensor()(image)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            keypoints = self.pose_estimator([x])
        return keypoints[0]
    
    @staticmethod
    def get_keypoints(dictionary, label=1, score_threshold=0.5):
        """
        Get Keypoints from the model Predictions

        Parameters
        ----------
        dictionary : Python dictionary, default=None
            Dictionary containing model predictions

        label : integer, default=1
            Keypoint label

        score_threshold : float, default=5
            Keypoint score threshold to be considered
        """

        keypoints = []
        if dictionary:           
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    keypoint = dictionary['keypoints'][i].detach().cpu().squeeze().numpy()
                    keypoints.append(keypoint)
        return np.asarray(keypoints, dtype=np.int32)
    