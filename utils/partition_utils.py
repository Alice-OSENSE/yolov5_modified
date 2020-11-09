import numpy as np
import yaml
from utils.general import rotate_bbox


class Segmenter:
    def __init__(self, config_path):
        self.in_use = config_path is not None
        if not self.in_use:
            return

        with open(config_path) as file:
            self.segment_dict = yaml.load(file)
            self.segment_dict['rot90'] = [i % 4 for i in self.segment_dict['rot90']]
        self.n_segments = len(self.segment_dict['subimages'])

    def __call__(self, img, channel_last=True):
        if not self.in_use:
            return [img]
        segments = [self._slice(img=img, index=i, channel_last=channel_last) for i in range(self.n_segments)]
        segments = [np.rot90(subimg, k=k).copy() for subimg, k in zip(segments, self.segment_dict['rot90'])]

        return segments

    def get_offset(self, img, subimg_index):
        # TODO: also return the offset of the subimage's upper left corner w.r.t. the original image
        """
        y = axis 0 (horizontal)
        x = axis 1 (vertical)
        """
        subimg = self.segment_dict['subimages'][subimg_index]
        offset = [int(subimg[0] * img.shape[1]), int(subimg[1] * img.shape[0])]
        return offset

    def get_subimage_wh(self, img_shape, subimg_index, channel_last=True):
        xyxy = self.get_subimage_xyxy(img_shape, subimg_index, channel_last=channel_last)
        return xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]

    def get_subimage_xyxy(self, img_shape, subimg_index, channel_last=True):
        start_axis = 0
        if not channel_last:
            start_axis = 1

        def get_valid_img_index(num, axis):
            return round(max(0.0, min(img_shape[axis], num)))

        subimage = self.segment_dict['subimages'][subimg_index]
        x1 = get_valid_img_index(subimage[0] * img_shape[start_axis], start_axis)
        y1 = get_valid_img_index(subimage[1] * img_shape[start_axis + 1], start_axis + 1)
        x2 = get_valid_img_index((subimage[0] + subimage[2]) * img_shape[start_axis], start_axis)
        y2 = get_valid_img_index((subimage[1] + subimage[3]) * img_shape[start_axis + 1], start_axis + 1)
        return [x1, y1, x2, y2]

    def _slice(self, img, index, channel_last=True):
        # we may assume img is 3-dimensional
        xyxy = self.get_subimage_xyxy(img.shape, index, channel_last=channel_last)
        if channel_last:
            return img[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3], :]
        else:
            return img[:, xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]]

    def get_offset_fraction(self, subimg_index):
        subimage_coordinates = self.segment_dict['subimages'][subimg_index]
        return subimage_coordinates[0], subimage_coordinates[1]

    def put_in_place(self, subimg_index, subimg_size, xyxy):
        return rotate_bbox(subimg_size, xyxy, k=-self.segment_dict['rot90'][subimg_index])