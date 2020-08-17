import cv2
import numpy as np


def _draw_keypoint(image, point, color, radius=1):
    x, y, v = point
    if int(v):
        cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return image


def _draw_connection(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if int(v1) and int(v2):
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image


def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for p in kp:
            overlay = _draw_keypoint(overlay, p, (0, 255, 0), radius)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_body_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    b_conn = [(0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    h_conn = [(0, 1), (0, 2), (1, 3), (2, 4)]
    l_conn = [(5, 7), (7, 9), (11, 13), (13, 15)]
    r_conn = [(6, 8), (8, 10), (12, 14), (14, 16)]
    for kp in keypoints:
        for i, j in b_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in h_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in l_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in r_conn:
            overlay = _draw_connection(overlay, kp[i], kp[j], (255, 0, 255), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

