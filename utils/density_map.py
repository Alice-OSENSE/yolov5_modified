import numpy as np


def multivariate_gaussian(pos, mean, cov, dim=2):
    cov_det = np.linalg.det(cov)
    print(cov_det)
    cov_inv = np.linalg.inv(cov)
    print(cov_inv)
    N = np.sqrt((2 * np.pi) ** dim * cov_det)
    factor = np.einsum('...k, kl, ...l->...', pos-mean, cov_inv, pos-mean)
    return np.exp((-factor / 2)) / N


def plot_one_density_distribution(x, pos, dmap=None):
    """
    x (four element float array) = (x, y, w, h)
    foreground_mask
    dmap (density map)
    """
    mean = np.array([int(x[0][0] + x[0][2] / 2), int(x[0][1] + x[0][3] / 2)])
    # Should depend not only on the bbox but also the shape of the foreground_mask
    cov = np.array([[x[0][2] / 6, 0.01],
                    [0.01,     x[0][3] / 6]])

    new_gaussian = multivariate_gaussian(pos=pos, mean=mean, cov=cov)
    if dmap is not None:
        return np.add(dmap, new_gaussian)

    # return new_gaussian