import cv2
import numpy as np
from skimage.transform import match_histograms


def get_foreground_mask(img, background, blur_size, dilation_kernel, threshold=20):
    # both img and background should have the same number of channels!
    difference = np.abs(img.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)

    blurred = cv2.GaussianBlur(difference, ksize=(blur_size, blur_size), sigmaX=3, sigmaY=3)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, kernel=dilation_kernel)
    return mask


def match_color(img, reference):
    """
    Adjust img_2's color to better match that of img_1's
    """
    matched = match_histograms(img, reference)
    return matched
