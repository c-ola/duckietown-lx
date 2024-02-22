import cv2
import numpy as np
    
#lower_hsv = np.array([0, 41, 55])
#upper_hsv = np.array([84, 255, 255])

lower_hsv = np.array([10, 39, 84])
upper_hsv = np.array([94, 255, 255])

def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Returns a 2D array"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask
