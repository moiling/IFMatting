import cv2
import numpy as np
import scipy

import os
import matplotlib.pyplot as plt


def mul_matT_mat(A, B):
    # calculates A.T B
    return np.einsum("...ji,...jk->...ik", A, B)


def mul_mat_mat_matT(A, B, C):
    # calculates A B C.T
    return np.einsum("...ij,...jk,...lk->...il", A, B, C)


def vec_vec_outer(a, b):
    return np.einsum("...i,...j", a, b)


def mul_vec_mat_vec(v, A, w):
    # calculates v' A w
    return np.einsum("...i,...ij,...j->...", v, A, w)


def mul_mat_matT(A, B):
    # calculates A B.T
    return np.einsum("...ij,...kj->...ik", A, B)


def save_image(image, save_dir, file_name, grey=False):
    # mkdir and touch
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + file_name):
        os.system(r"touch {}".format(save_dir + file_name))

    if grey:
        cv2.imwrite(save_dir + file_name, (image * 255).astype(np.int))
        # plt.imsave(save_dir + file_name, image, cmap='Greys_r')
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


def local_linear_embedding(pt, neighbors, conditioner_mult):
    corr = neighbors.dot(neighbors.T) + np.eye(neighbors.shape[0]) * conditioner_mult

    pt_dot_n = neighbors.T.dot(pt)
    alpha = 1 - np.sum(np.linalg.lstsq(corr, pt_dot_n, rcond=None)[0])  # 1 - sum(corr \ pt_dot_n)
    beta = np.sum(np.linalg.lstsq(corr, np.ones((corr.shape[0], 1)), rcond=None)[0])  # sum of elements of inv(corr)
    lagrange_mult = alpha / beta
    w = np.linalg.lstsq(corr, (pt_dot_n + lagrange_mult), rcond=None)[0]
    return w


def affinity_matrix_to_laplacian(A):
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


def make_windows(image, radius=1):
    return np.stack([image[
                     y:y + image.shape[0] - 2 * radius,
                     x:x + image.shape[1] - 2 * radius]
                     for x in range(2 * radius + 1)
                     for y in range(2 * radius + 1)],
                    axis=2)


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


def solve_cg(
    A,
    b,
    rtol,
    max_iter,
    atol=0.0,
    x0=None,
    precondition=None,
    callback=None,
    print_info=False,
):
    """
    Solve the linear system Ax = b for x using preconditioned conjugate
    gradient descent.

    A: np.ndarray of dtype np.float64
        Must be a square symmetric matrix
    b: np.ndarray of dtype np.float64
        Right-hand side of linear system
    rtol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < relative_tolerance norm(b)
    max_iter: int
        Maximum number of iterations
    atol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < absolute_tolerance
    x0: np.ndarray of dtype np.float64
        Initial guess of solution x
    precondition: func(r) -> r
        Improve solution of residual r, for example solve(M, r)
        where M is an easy-to-invert approximation of A.
    callback: func(A, x, b)
        callback to inspect temporary result after each iteration.
    print_info: bool
        If to print convergence information.

    Returns
    -------

    x: np.ndarray of dtype np.float64
        Solution to the linear system Ax = b.

    """

    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()

    if callback is not None:
        callback(A, x, b)

    if precondition is None:
        def precondition(r):
            return r

    norm_b = np.linalg.norm(b)

    r = b - A.dot(x)
    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)
    for iteration in range(max_iter):
        Ap = A.dot(p)
        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        residual_error = np.linalg.norm(r)

        if print_info:
            print("iteration %05d - residual error %e" % (
                iteration,
                residual_error))

        if callback is not None:
            callback(A, x, b)

        if residual_error < atol or residual_error < rtol * norm_b:
            break

        z = precondition(r)
        beta = 1.0 / rz
        rz = np.inner(r, z)
        beta *= rz
        p *= beta
        p += z

    return x