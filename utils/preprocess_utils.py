import cv2
import numpy as np
from skimage import exposure
from skimage.transform import match_histograms



def get_foreground_mask(img, background, dilation_kernel, threshold=20):
    difference = np.abs(img.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)
    blurred = cv2.GaussianBlur(difference, ksize=(5, 5), sigmaX=3, sigmaY=3)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, kernel=dilation_kernel)
    return mask


def match_color(img, reference):
    """
    Adjust img_2's color to better match that of img_1's
    """
    matched = match_histograms(img, reference)
    return matched
