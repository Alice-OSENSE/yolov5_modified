import numpy as np
from matplotlib import pyplot as plt

from utils.density_map_utils import plot_one_density_distribution, setup_density_map

if __name__ == '__main__':
    bbox = [[50, 50., 10., 20.]]
    img_size = [100, 120]
    img = np.zeros(img_size)
    dmap, pos = setup_density_map(img)
    dmap = plot_one_density_distribution(bbox, pos, dmap)
    plt.imshow(dmap)
    plt.show()