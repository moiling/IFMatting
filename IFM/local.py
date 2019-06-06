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

    # dilUnk = cv2.dilate(is_unknown, np.ones((window_radius, window_radius), np.uint8))

    dilUnk = imdilate(is_unknown, window_radius)

    window_size = (2 * window_radius + 1) ** 2

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(height, width, 1, depth)
    # shape: h w 3 3
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / window_size

    inv_cov = np.linalg.inv(covariance + epsilon / window_size * np.eye(3, 3))

    indices = np.arange(width * height).reshape(height, width)
    neighInd = make_windows(indices)

    inMap = dilUnk[window_radius:-window_radius, window_radius:-window_radius]

    neighInd = neighInd.reshape(-1, window_size)

    neighInd = neighInd[inMap.flatten()]

    inInd = neighInd[:, window_size // 2]

    image = image.reshape(-1, 3)
    means = means.reshape(-1, 3)
    inv_cov = inv_cov.reshape(-1, 3, 3)

    centered_neighbors = image[neighInd] - means[inInd].reshape(-1, 1, 3)

    weights = mul_mat_mat_matT(centered_neighbors, inv_cov[inInd], centered_neighbors)

    flowCols = np.repeat(neighInd, window_size, axis=1).reshape(-1, window_size, window_size)
    flowRows = flowCols.transpose(0, 2, 1)

    weights = (weights + 1) / window_size

    flowRows = flowRows.flatten()
    flowCols = flowCols.flatten()
    weights = weights.flatten()

    W = csc_matrix((weights, (flowRows, flowCols)), shape=(n, n))

    W = W + W.T

    W_row_sum = np.array(W.sum(axis=1)).flatten()
    W_row_sum[W_row_sum < 0.05] = 1.0

    return diags(1 / W_row_sum).dot(W)
