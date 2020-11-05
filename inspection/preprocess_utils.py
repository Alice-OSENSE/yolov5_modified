import cv2
import numpy as np
from skimage.transform import match_histograms

from utils.preprocess_utils import get_foreground_mask

if __name__ == '__main__':
    background_path = '/home/osense-office/Desktop/background.jpg'
    foreground_path = '/home/osense-office/Desktop/foreground.jpg'  # TODO: retrieve foreground frame
    background = cv2.imread(background_path)
    foreground = cv2.imread(foreground_path)

    # matched = match_histograms(foreground, background, multichannel=True)
    # matched = cv2.resize(matched, None, fx=0.4, fy=0.4)

    img = get_foreground_mask(foreground, background, dilation_kernel=np.ones((5,5)), threshold=50)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    cv2.imshow("matching result", img)
    cv2.waitKey(0)
    # cv2.imwrite('/home/osense-office/Desktop/foreground_matched.jpg', matched)
