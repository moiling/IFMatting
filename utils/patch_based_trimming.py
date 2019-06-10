import numpy as np

from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.utils import make_windows, pad, mul_matT_mat, mul_vec_mat_vec


def patch_based_trimming(image, trimap, minDist, maxDist, windowRadius, K):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    trimap = trimap.copy()
    height, width, depth = image.shape

    eps = 1e-8

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(height, width, 1, depth)
    # shape: h w 3 3
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / (3 * 3) + eps / (3 * 3) * np.eye(3, 3)

    unk_ind, fg_neigh = find_non_local_neighbors(means, K, None, is_unknown, is_fg)
    _, bg_neigh = find_non_local_neighbors(means, K, None, is_unknown, is_bg)

    mean_image = means.transpose(0, 1, 2).reshape(height * width, depth)

    covariance = covariance.transpose(0, 1, 2, 3).reshape(width * height, 3, 3)

    pix_means = mean_image[unk_ind]
    pix_covars = covariance[unk_ind]
    pix_dets = np.linalg.det(pix_covars)
    pix_covars = pix_covars.reshape(unk_ind.shape[0], 1, 3, 3)

    n_means = mean_image[fg_neigh] - pix_means.reshape(unk_ind.shape[0], 1, 3)
    n_covars = covariance[fg_neigh]
    n_dets = np.linalg.det(n_covars)
    n_covars = (pix_covars + n_covars) / 2

    fg_bhatt = 0.125 * mul_vec_mat_vec(n_means, np.linalg.inv(n_covars), n_means) + 0.5 * np.log(
        np.linalg.det(n_covars) / np.sqrt(pix_dets[:, None] * n_dets))

    n_means = mean_image[bg_neigh] - pix_means.reshape(unk_ind.shape[0], 1, 3)
    n_covars = covariance[bg_neigh]
    n_dets = np.linalg.det(n_covars)
    n_covars = (pix_covars + n_covars) / 2

    bg_bhatt = 0.125 * mul_vec_mat_vec(n_means, np.linalg.inv(n_covars), n_means) + 0.5 * np.log(
        np.linalg.det(n_covars) / np.sqrt(pix_dets[:, None] * n_dets))

    shape = trimap.shape

    min_f_gdist = np.min(fg_bhatt, axis=1)
    min_b_gdist = np.min(bg_bhatt, axis=1)

    mask0 = np.logical_and(min_b_gdist < minDist, min_f_gdist > maxDist)
    mask1 = np.logical_and(min_f_gdist < minDist, min_b_gdist > maxDist)

    trimap[np.unravel_index(unk_ind[mask0], shape)] = 0
    trimap[np.unravel_index(unk_ind[mask1], shape)] = 1

    return trimap
