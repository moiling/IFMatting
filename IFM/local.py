#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 18:53
# @Author  : moiling
# @File    : local.py
import cv2
import numpy as np
from scipy.sparse import csr_matrix, spdiags, csc_matrix, diags

from utils.closed_form_matting import compute_weight
from utils.utils import imdilate, make_windows, pad, mul_matT_mat, mul_mat_mat_matT, local_rgb_normal_distributions, \
    im2col


def local_3(image, trimap):
    umask = (trimap != 0) & (trimap != 1)
    w_l = compute_weight(image, mask=umask).tocsr()
    return w_l


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


def local_2(image, trimap, window_radius=1, epsilon=1e-7):
    in_map = trimap.copy()
    in_map[(trimap != 1) & (trimap != 0)] = 1
    in_map[(trimap == 1) | (trimap == 0)] = 0
    in_map = cv2.dilate(
        in_map,
        np.ones(2 * window_radius + 1)
    )
    window_size = 2 * window_radius + 1
    neighbors_size = window_size ** 2
    h, w, c = image.shape
    n = h * w
    epsilon = epsilon / neighbors_size

    mean_image, covar_mat = local_rgb_normal_distributions(image, window_radius, epsilon)

    # Determine pixels and their local neighbors
    indices = np.arange(h * w).reshape((h, w))
    neighbors_indices = im2col(indices, [window_size, window_size])
    in_map = in_map[window_radius + 1: -window_radius, window_radius + 1: -window_radius]

    neighbors_indices = neighbors_indices[in_map != 0, :]
    in_indices = neighbors_indices[:, (neighbors_size + 1) / 2]
    pix_count = in_indices.shape[0]

    # Prepare in & out data
    image = image.reshape((n, c))
    mean_image = mean_image.reshape((n, c))
    flow_rows = np.zeros((neighbors_size, neighbors_size, pix_count))
    flow_cols = np.zeros((neighbors_size, neighbors_size, pix_count))
    flows = np.zeros((neighbors_size, neighbors_size, pix_count))

    # Compute matting affinity
    for i in range(indices.shape[0]):
        neighbors = neighbors_indices[i, :]
        shifted_window_colors = image[neighbors, :] - np.tile(mean_image[in_indices[i], :], neighbors.shape[1])
        flows[:, :, i] = shifted_window_colors * (np.linalg.lstsq(covar_mat[:, :, in_indices[i]], shifted_window_colors.T))
        neighbors = np.tile(neighbors, neighbors.shape[1])
        flow_rows[:, :, 1] = neighbors
        flow_cols[:, :, 1] = neighbors.T

    flows = (flows + 1) / neighbors_size
    w = csr_matrix((flows.flatten(), (flow_rows.flatten(), flow_cols.flatten())), shape=(n, n))

    # Make sure it's symmetric
    w = w + w.T

    # Normalize
    sum_w = np.sum(w.A, 1)
    sum_w[sum_w < 0.05] = 1
    w = spdiags(1 / sum_w.flatten(), 0, n, n) * w
    return w
