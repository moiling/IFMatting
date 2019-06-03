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
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / (3 * 3) \
        + eps / (3 * 3) * np.eye(3, 3)

    unkInd, fgNeigh = find_non_local_neighbors(means, K, None, is_unknown, is_fg)
    _, bgNeigh = find_non_local_neighbors(means, K, None, is_unknown, is_bg)

    meanImage = means.transpose(0, 1, 2).reshape(height * width, depth)

    covariance = covariance.transpose(0, 1, 2, 3).reshape(width * height, 3, 3)

    pixMeans = meanImage[unkInd]
    pixCovars = covariance[unkInd]
    pixDets = np.linalg.det(pixCovars)
    pixCovars = pixCovars.reshape(unkInd.shape[0], 1, 3, 3)

    nMeans = meanImage[fgNeigh] - pixMeans.reshape(unkInd.shape[0], 1, 3)
    nCovars = covariance[fgNeigh]
    nDets = np.linalg.det(nCovars)
    nCovars = (pixCovars + nCovars) / 2

    fgBhatt = 0.125 * mul_vec_mat_vec(nMeans, np.linalg.inv(nCovars), nMeans) \
        + 0.5 * np.log(np.linalg.det(nCovars) / np.sqrt(pixDets[:, None] * nDets))

    nMeans = meanImage[bgNeigh] - pixMeans.reshape(unkInd.shape[0], 1, 3)
    nCovars = covariance[bgNeigh]
    nDets = np.linalg.det(nCovars)
    nCovars = (pixCovars + nCovars) / 2

    bgBhatt = 0.125 * mul_vec_mat_vec(nMeans, np.linalg.inv(nCovars), nMeans) \
        + 0.5 * np.log(np.linalg.det(nCovars) / np.sqrt(pixDets[:, None] * nDets))

    shape = trimap.shape

    minFGdist = np.min(fgBhatt, axis=1)
    minBGdist = np.min(bgBhatt, axis=1)

    mask0 = np.logical_and(minBGdist < minDist, minFGdist > maxDist)
    mask1 = np.logical_and(minFGdist < minDist, minBGdist > maxDist)

    trimap[np.unravel_index(unkInd[mask0], shape)] = 0
    trimap[np.unravel_index(unkInd[mask1], shape)] = 1

    return trimap
