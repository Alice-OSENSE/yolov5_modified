import numpy as np
import yaml


class Segmenter:
    def __init__(self, config_path):
        with open(config_path) as file:
            self.segment_dict = yaml.load(file)
            self.segment_dict['rot90'] = [i % 4 for i in self.segment_dict['rot90']]
        self.n_segments = len(self.segment_dict['subimages'])

    def __call__(self, img):
        segments = [self._slice(img=img, index=i) for i in range(self.n_segments)]
        segments = [np.rot90(img, k=k) for img, k in zip(segments, self.segment_dict['rot90'])]
        return segments

    def _slice(self, img, index, channel_last=True):
        img_shape = img.shape
        # we may assume img is 3-dimensional
        start_index = 0
        if not channel_last:
            start_index = 1

        def get_valid_img_index(num, axis):
            return round(min(0.0, max(img_shape[axis], num)))

        subimage = self.segment_dict['subimages'][index]
        x1 = get_valid_img_index(subimage[0] * img_shape[start_index])
        y1 = get_valid_img_index(subimage[1] * img_shape[start_index+1])
        x2 = get_valid_img_index((subimage[0] + subimage[2]) * img_shape[start_index])
        y2 = get_valid_img_index((subimage[1] + subimage[3]) * img_shape[start_index + 1])

        if channel_last:
            return img[x1:x2, y1:y2, :]
        else:
            return img[:, x1:x2, y1:y2]