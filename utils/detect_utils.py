import numpy as np

def prepare_for_density_map(img):
    dmap = np.zeros([img.shape[0], img.shape[1], 1])
    x_axis = np.linspace(0, img.shape[0], img.shape[0])
    y_axis = np.linspace(0, img.shape[1], img.shape[1])
    x, y = np.meshgrid(x_axis, y_axis)

    # Pack x and y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    return dmap, pos, None, None
