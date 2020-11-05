import numpy as np

def setup_density_map(img):
    dmap = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.float32)

    x_axis = np.linspace(0, img.shape[1], img.shape[1], dtype=np.float32)
    y_axis = np.linspace(0, img.shape[0], img.shape[0], dtype=np.float32)
    x, y = np.meshgrid(x_axis, y_axis)

    # Pack x and y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,), dtype=np.float32)
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    print("dmap pos")
    print(dmap.shape)
    print(pos.shape)
    return dmap, pos


def multivariate_gaussian(pos, mean, cov, dim=2):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    N = np.sqrt((2 * np.pi) ** dim * cov_det)
    factor = np.einsum('...k, kl, ...l->...', pos-mean, cov_inv, pos-mean)
    return np.exp((-factor / 2)) / N


def plot_one_density_distribution(x, pos, dmap=None):
    """
    x (four element float array) = (x, y, x, y)
    foreground_mask TODO: incorporate foreground mask to make the density map more realistic
    dmap (density map)
    """
    mean = np.array([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2]).astype(np.float32)
    # Should depend not only on the bbox but also the shape of the foreground_mask
    cov = np.array([[(x[2] - x[0]) * 30., 0.01],
                    [0.01,     (x[3] - x[1]) * 30.]]).astype(np.float32)

    new_gaussian = multivariate_gaussian(pos=pos, mean=mean, cov=cov)
    new_gaussian = np.expand_dims(new_gaussian, axis=2) * 500

    if dmap is not None:
        return np.add(dmap, new_gaussian)
    else:
        return new_gaussian