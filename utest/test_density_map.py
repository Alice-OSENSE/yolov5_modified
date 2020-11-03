import numpy as np
from matplotlib import pyplot as plt

from utils.density_map import plot_one_density_distribution

if __name__ == '__main__':
    bbox = [0., 0., 1, 2]
    img_size = [100, 100]
    x_axis = np.linspace(0, img_size[0], img_size[0])
    y_axis = np.linspace(0, img_size[1], img_size[1])
    x, y = np.meshgrid(x_axis, y_axis)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    dmap = np.zeros(img_size)
    result = plot_one_density_distribution(bbox, pos, dmap)
    plt.imshow(result)
    plt.show()