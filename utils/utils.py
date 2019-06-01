import cv2
import numpy as np
import scipy

from IFM.find_non_local_neighbors import find_non_local_neighbors

import os
import matplotlib.pyplot as plt


def save_image(image, save_dir, file_name, grey=False):
    # mkdir and touch
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + file_name):
        os.system(r"touch {}".format(save_dir + file_name))

    if grey:
        plt.imsave(save_dir + file_name, image, cmap='Greys_r')
    else:
        print(image.shape)
        plt.imsave(save_dir + file_name, image)


def show_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def stack_alpha(image, alpha):
    if len(alpha.shape) == 3:
        alpha = alpha[:, :, 0]

    return np.concatenate([image, alpha[:, :, np.newaxis]], axis=2)


def stack_images(*images):
    return np.concatenate([
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ], axis=2)


def pixel_coordinates(w, h, flat=False):
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    if flat:
        x = x.flatten()
        y = y.flatten()

    return x, y


def vec_vec_outer(a, b):
    return np.einsum("...i,...j", a, b)


def inv2(mat):
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]

    inv_det = 1 / (a * d - b * c)

    inv = np.empty(mat.shape)

    inv[..., 0, 0] = inv_det * d
    inv[..., 0, 1] = inv_det * -b
    inv[..., 1, 0] = inv_det * -c
    inv[..., 1, 1] = inv_det * a

    return inv


def resize_nearest(image, new_width, new_height):
    old_height, old_width = image.shape[:2]

    x = np.arange(new_width)[np.newaxis, :]
    y = np.arange(new_height)[:, np.newaxis]
    x = x * old_width / new_width
    y = y * old_height / new_height
    x = np.clip(x.astype(np.int32), 0, old_width - 1)
    y = np.clip(y.astype(np.int32), 0, old_height - 1)

    if len(image.shape) == 3:
        image = image.reshape(-1, image.shape[2])
    else:
        image = image.ravel()

    return image[x + y * old_width]


def resize_bilinear(image, new_width, new_height):
    old_height, old_width = image.shape[:2]

    x2 = old_width / new_width * (np.arange(new_width) + 0.5) - 0.5
    y2 = old_height / new_height * (np.arange(new_height) + 0.5) - 0.5

    x2 = x2[np.newaxis, :]
    y2 = y2[:, np.newaxis]

    x0 = np.floor(x2)
    y0 = np.floor(y2)

    ux = x2 - x0
    uy = y2 - y0

    x0 = x0.astype(np.int32)
    y0 = y0.astype(np.int32)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, old_width - 1)
    x1 = np.clip(x1, 0, old_width - 1)

    y0 = np.clip(y0, 0, old_height - 1)
    y1 = np.clip(y1, 0, old_height - 1)

    if len(image.shape) == 3:
        pix = image.reshape(-1, image.shape[2])
        ux = ux[..., np.newaxis]
        uy = uy[..., np.newaxis]
    else:
        pix = image.ravel()

    a = (1 - ux) * pix[y0 * old_width + x0] + ux * pix[y0 * old_width + x1]
    b = (1 - ux) * pix[y1 * old_width + x0] + ux * pix[y1 * old_width + x1]

    return (1 - uy) * a + uy * b


def mul_vec_mat_vec(v, A, w):
    # calculates v' A w
    return np.einsum("...i,...ij,...j->...", v, A, w)


def patchBasedTrimming(image, trimap, minDist, maxDist, windowRadius, K):
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


def solve_for_weights(z, regularization_factor=1e-3):
    n, n_neighbors, _ = z.shape

    # calculate covariance matrices
    C = mul_mat_matT(z, z)

    # regularization
    C += regularization_factor * np.eye(n_neighbors)

    # solve for weights
    weights = np.linalg.solve(C, np.ones((n, n_neighbors)))
    # normalize rows
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def mul_mat_matT(A, B):
    # calculates A B.T
    return np.einsum("...ij,...kj->...ik", A, B)


def affinityMatrixToLaplacian(A):
    return weights_to_laplacian(A, normalize=False)


def weights_to_laplacian(W, normalize=True):
    if normalize:
        # normalize row sum to 1
        W_row_sum = np.array(W.sum(axis=1)).flatten()
        W_row_sum[np.abs(W_row_sum) < 1e-10] = 1e-10
        W = scipy.sparse.diags(1 / W_row_sum).dot(W)
        L = scipy.sparse.identity(len(W_row_sum)) - W
    else:
        W_row_sum = np.array(W.sum(axis=1)).flatten()
        L = scipy.sparse.diags(W_row_sum) - W
    return L


def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result


def make_windows(image, radius=1):
    return np.stack([image[
        y:y + image.shape[0] - 2 * radius,
        x:x + image.shape[1] - 2 * radius]
        for x in range(2 * radius + 1)
        for y in range(2 * radius + 1)],
        axis=2)


def mul_matT_mat(A, B):
    # calculates A.T B
    return np.einsum("...ji,...jk->...ik", A, B)


def mul_mat_mat_matT(A, B, C):
    # calculates A B C.T
    return np.einsum("...ij,...jk,...lk->...il", A, B, C)


def pad(image, r=1):
    # pad by repeating border pixels of image

    # create padded result image with same shape as input
    if len(image.shape) == 2:
        height, width = image.shape
        padded = np.zeros((height + 2 * r, width + 2 * r), dtype=image.dtype)
    else:
        height, width, depth = image.shape
        padded = np.zeros((height + 2 * r, width + 2 * r, depth), dtype=image.dtype)

    # do padding
    if r > 0:
        # top left
        padded[:r, :r] = image[0, 0]
        # bottom right
        padded[-r:, -r:] = image[-1, -1]
        # top right
        padded[:r, -r:] = image[0, -1]
        # bottom left
        padded[-r:, :r] = image[-1, 0]
        # left
        padded[r:-r, :r] = image[:, :1]
        # right
        padded[r:-r, -r:] = image[:, -1:]
        # top
        padded[:r, r:-r] = image[:1, :]
        # bottom
        padded[-r:, r:-r] = image[-1:, :]

    # center
    padded[r:r + height, r:r + width] = image

    return padded


def imdilate(image, radius):
    return boxfilter(image, radius, 'same') > 0


def boxfilter(image, r, mode='full', fill_value=0.0):
    height, width = image.shape[:2]
    size = 2 * r + 1

    pad = {
        'full': 2 * r,
        'valid': 0,
        'same': r,
    }[mode]

    shape = [1 + pad + height + pad, 1 + pad + width + pad] + list(image.shape[2:])
    image_padded = np.full(shape, fill_value, dtype=image.dtype)
    image_padded[1 + pad:1 + pad + height, 1 + pad:1 + pad + width] = image

    c = np.cumsum(image_padded, axis=0)
    c = c[size:, :] - c[:-size, :]
    c = np.cumsum(c, axis=1)
    c = c[:, size:] - c[:, :-size]

    return c


def local_rgb_normal_distributions(image, window_radius, epsilon):
    h, w, _ = image.shape
    n = h * w
    window_size = 2 * window_radius + 1
    mean_image = cv2.boxFilter(image, cv2.CV_64F, (window_size, window_size))
    covar_mat = np.zeros((3, 3, n))

    for r in range(3):
        for c in range(3):
            temp = cv2.boxFilter(image[:, :, r] * image[:, :, c], cv2.CV_64F, (window_size, window_size)) \
                   - mean_image[:, :, r] * mean_image[:, :, c]
            covar_mat[r, c, :] = temp.flatten()

    for i in range(3):
        covar_mat[i, i, :] = covar_mat[i, i, :] + epsilon

    for r in range(1, 3):
        for c in range(r - 1):
            covar_mat[r, c, :] = covar_mat[c, r, :]

    return mean_image, covar_mat
