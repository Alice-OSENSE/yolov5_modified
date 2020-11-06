import argparse
import cv2
import numpy as np
from pathlib import Path
from skimage.transform import match_histograms

from utils.partition_utils import Segmenter


def get_foreground_mask(img, background, dilation_kernel, threshold=20):
    difference = np.abs(img.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)
    blurred = cv2.GaussianBlur(difference, ksize=(5, 5), sigmaX=3, sigmaY=3)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, kernel=dilation_kernel)
    return mask


def inspect_foreground_extraction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fg_path', type=str, help='foreground path')
    parser.add_argument('--bg_path', type=str, help='background path')
    parser.add_argument('--save_path', type=str, help='The path to save the result')
    opt = parser.parse_args()

    background_path = Path(opt.bg_path)
    foreground_path = Path(opt.fg_path)
    background = cv2.imread(opt.bg_path)
    foreground = cv2.imread(opt.fg_path)

    # matched = match_histograms(foreground, background, multichannel=True)
    # matched = cv2.resize(matched, None, fx=0.4, fy=0.4)

    mask = get_foreground_mask(foreground, background, dilation_kernel=np.ones((20, 20)), threshold=40)
    # mask = cv2.resize(mask, None, fx=0.4, fy=0.4)
    mask = np.expand_dims(np.amin(mask, axis=2), axis=2)
    masked_fg = foreground
    masked_fg[:, :, 1:] = cv2.bitwise_and(foreground[:, :, 1:], foreground[:, :, 1:], mask=mask)
    masked_fg = cv2.resize(masked_fg, None, fx=0.4, fy=0.4)
    cv2.imshow("matching result", masked_fg)
    cv2.waitKey(0)
    cv2.imwrite(opt.save_path, mask)


def inspect_segmenter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to config files for segmenter')
    parser.add_argument('--img_path', type=str, help='Path to the image')
    opt = parser.parse_args()

    img = cv2.imread(opt.img_path)
    segmenter = Segmenter(config_path=opt.config_path)
    sub_images = segmenter(img)
    for img in sub_images:
        cv2.imshow("img", img)
        cv2.waitKey(0)

if __name__ == '__main__':
    inspect_segmenter()