import numpy as np
from scipy.sparse import csc_matrix, diags

from utils.utils import imdilate, make_windows, pad, mul_matT_mat, mul_mat_mat_matT


def local(image, trimap, window_radius=1, epsilon=1e-7):
    height, width, depth = image.shape
    n = height * width
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    dil_unk = imdilate(is_unknown, window_radius)

    window_size = (2 * window_radius + 1) ** 2

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(height, width, 1, depth)
    # shape: h w 3 3
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / window_size

    inv_cov = np.linalg.inv(covariance + epsilon / window_size * np.eye(3, 3))

    indices = np.arange(width * height).reshape(height, width)
    neigh_ind = make_windows(indices)

    in_map = dil_unk[window_radius:-window_radius, window_radius:-window_radius]

    neigh_ind = neigh_ind.reshape(-1, window_size)

    neigh_ind = neigh_ind[in_map.flatten()]

    in_ind = neigh_ind[:, window_size // 2]

    image = image.reshape(-1, 3)
    means = means.reshape(-1, 3)
    inv_cov = inv_cov.reshape(-1, 3, 3)

    centered_neighbors = image[neigh_ind] - means[in_ind].reshape(-1, 1, 3)

    weights = mul_mat_mat_matT(centered_neighbors, inv_cov[in_ind], centered_neighbors)

    flow_cols = np.repeat(neigh_ind, window_size, axis=1).reshape(-1, window_size, window_size)
    flow_rows = flow_cols.transpose(0, 2, 1)

    weights = (weights + 1) / window_size

    flow_rows = flow_rows.flatten()
    flow_cols = flow_cols.flatten()
    weights = weights.flatten()

    W = csc_matrix((weights, (flow_rows, flow_cols)), shape=(n, n))

    W = W + W.T

    W_row_sum = np.array(W.sum(axis=1)).flatten()
    W_row_sum[W_row_sum < 0.05] = 1.0

    return diags(1 / W_row_sum).dot(W)
