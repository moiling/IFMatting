# The information flow alpha matting method implementation in this file
# is based on
# https://github.com/yaksoy/AffinityBasedMattingToolbox
# by Yağız Aksoy.
#
############################################################################################
# Copyright 2017, Yagiz Aksoy. All rights reserved.                                        #
#                                                                                          #
# This software is for academic use only. A redistribution of this                         #
# software, with or without modifications, has to be for academic                          #
# use only, while giving the appropriate credit to the original                            #
# authors of the software. The methods implemented as a part of                            #
# this software may be covered under patents or patent applications.                       #
#                                                                                          #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR IMPLIED             #
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND #
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR         #
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR      #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF     #
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                               #
############################################################################################

r"""
The information flow alpha matting method is provided for academic use only.
If you use the information flow alpha matting method for an academic
publication, please cite corresponding publications referenced in the
description of each function:

@INPROCEEDINGS{ifm,
author={Aksoy, Ya\u{g}{\i}z and Ayd{\i}n, Tun\c{c} Ozan and Pollefeys, Marc},
booktitle={Proc. CVPR},
title={Designing Effective Inter-Pixel Information Flow for Natural Image Matting},
year={2017},
}
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.neighbors import KDTree

from scipy.sparse import csr_matrix as csr, diags
from closed_form_matting import compute_weight

from lle import barycenter_kneighbors_graph as bkg
from lle import barycenter_kneighbors_graph_ku as bkg_ku
from color_mixture import color_mixture as cm
from intra_u import intra_u as uu


def color_mixture(k, feature):

    # use barycenter k neighbors graph to compute (1)
    return bkg(feature, n_neighbors=k)


def k_to_u(alpha, k, feature):

    # pixel size
    n = feature.shape[0]

    index = np.arange(n)

    fore = feature[alpha == 1]
    fore_index = index[alpha == 1]

    back = feature[alpha == 0]
    back_index = index[alpha == 0]

    unknown = feature[(alpha != 0) & (alpha != 1)]
    unknown_index = index[(alpha != 0) & (alpha != 1)]

    # nearest foreground pixel to unknown
    kd_tree = KDTree(fore, leaf_size=30, metric='euclidean')
    # nf 就是一个(n, 7)矩阵，每行表示找出的7个临近点，n是未知点个数，所以nf、nb的n是相同的
    nf = kd_tree.query(unknown, k=k, return_distance=False)
    # fore_index是一个n维向量，把nf放进去，其实就是找每一个对应的index是什么，返回的还是(n, 7)矩阵
    nf_index = fore_index[nf]

    # nearest background pixel to unknown
    kd_tree = KDTree(back, leaf_size=30, metric='euclidean')
    nb = kd_tree.query(unknown, k=k, return_distance=False)
    nb_index = back_index[nb]

    # 把两个(n, 7)合并成(n, 14)
    nfb_index = np.concatenate((nf_index, nb_index), axis=1)

    # feature[:,:-2]是[x, 3]，x为像素总数，在[x, 3]里找下标对应[n, 14]，变成[n, 14, 3]
    nfb_color = feature[:, :-2][nfb_index]
    unknown_color = unknown[:, :-2]

    # compute (1)
    w_ku = bkg_ku(unknown_color, nfb_color, nfb_index, n_neighbors=2 * k)

    # 为了和颜色相乘，要加一维
    w_ku = w_ku.reshape((w_ku.shape[0], w_ku.shape[1], 1))

    nfb_weighted_color = w_ku * nfb_color

    # 前7个是前景的, 计算公式(6)
    cpf = np.sum(nfb_weighted_color[:, :k, :], axis=1) / np.sum(w_ku[:, :k], axis=1)
    cpb = np.sum(nfb_weighted_color[:, k:, :], axis=1) / np.sum(w_ku[:, k:], axis=1)

    w_f = np.zeros(n)
    w_f[unknown_index] = np.sum(w_ku[:, 0:k, :], axis=1)[:, 0]
    w_f[fore_index] = 1
    # 公式(7)
    n_p = np.sum((cpf - cpb) * (cpf - cpb), axis=1) / 3
    n_p = csr((n_p, (unknown_index, unknown_index)), shape=(n, n))
    return w_f, n_p


def local(img, trimap):
    umask = (trimap != 0) & (trimap != 1)
    w_l = compute_weight(img, mask=umask).tocsr()
    return w_l


def intra_u(alpha, k, feature):
    n = feature.shape[0]
    index = np.arange(n)

    unknown = feature[(alpha != 0) & (alpha != 1)]
    unknown_index = index[(alpha != 0) & (alpha != 1)]

    # nearest background pixel to unknown
    kd_tree = KDTree(unknown, leaf_size=30, metric='euclidean')
    nu = kd_tree.query(unknown, k=k, return_distance=False)
    unk_nbr_true_ind = unknown_index[nu]
    # TODO
    unk_nu_ind = np.asarray([int(i / k) for i in range(nu.shape[0] * nu.shape[1])])
    unk_nu_true_ind = unknown_index[unk_nu_ind]

    nbr = unknown[nu]
    nbr = np.swapaxes(nbr, 1, 2)
    unk = unknown.reshape((unknown.shape[0], unknown.shape[1], 1))

    x = nbr - unk
    x = np.abs(x)
    y = 1 - np.sum(x, axis=1)
    y[y < 0] = 0

    row = unk_nu_true_ind
    col = unk_nbr_true_ind.ravel()
    data = y.ravel()
    z = csr((data, (col, row)), shape=(n, n))
    w_uu = csr((data, (row, col)), shape=(n, n))
    w_uu = w_uu + z
    return w_uu


def eq1(w_cm, w_uu, w_l, n_p, T, a_k, w_f, params):
    d_cm = diags(csr.sum(w_cm, axis=1).A.ravel(), format='csr')
    d_uu = diags(csr.sum(w_uu, axis=1).A.ravel(), format='csr')
    d_l = diags(csr.sum(w_l, axis=1).A.ravel(), format='csr')

    l_ifm = csr.transpose(d_cm - w_cm).dot(d_cm - w_cm) + params['s_uu'] * (d_uu - w_uu) + params['s_l'] * (d_l - w_l)

    if params['use_k_u']:
        # (16)
        A = l_ifm + params['lambda'] * T + params['s_ku'] * n_p
        b = (params['lambda'] * T + params['s_ku'] * n_p).dot(w_f)
    else:
        # (19)
        A = l_ifm + params['lambda'] * T
        b = params['lambda'] * T * a_k

    # use preconditioned conjugate gradient to solve the linear systems
    solution = scipy.sparse.linalg.cg(A, b, x0=w_f, tol=1e-7, maxiter=2000, M=None, callback=None)
    return solution[0]
    # solution = solve_cg(A, b, 0, 2000, 1e-5, print_info=True)
    # solution = np.clip(solution, 0, 1)
    # return solution


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


def init_params(img, h, w):
    # compute feature vector
    feature_cm = []
    feature_ku = []
    feature_uu = []

    for i in range(h):
        for j in range(w):
            r, g, b = list(img[i, j, :])
            feature_cm.append([r, g, b, i / h, j / w])
            feature_ku.append([r, g, b, i / h * 10, j / w * 10])
            feature_uu.append([r, g, b, i / h / 20, j / w / 20])

    feature_cm = np.asarray(feature_cm)
    feature_ku = np.asarray(feature_ku)
    feature_uu = np.asarray(feature_uu)

    params = {
        'k_cm': 20,
        'k_ku': 7,
        'k_uu': 5,
        's_ku': 0.05,
        's_uu': 0.01,
        's_l': 1,
        'lambda': 100,
        'use_k_u': True
    }
    return params, feature_cm, feature_ku, feature_uu


def matting(image, trimap):
    print('Start matting.')

    h, w, _ = image.shape
    params, feature_cm, feature_ku, feature_uu = init_params(image, h, w)
    alpha = trimap.flatten()

    print('Color-mixture information flow.')
    # w_cm = color_mixture(params['k_cm'], feature_cm)
    w_cm = cm(image, trimap, params['k_cm'], feature_cm)

    # TODO 判断是否执行k_u
    params['use_k_u'] = False

    if params['use_k_u']:
        print('K-to-U information flow.')
        w_f, n_p = k_to_u(alpha, params['k_ku'], feature_ku)
        a_k = None
    else:
        w_f = None
        n_p = None
        a_k = alpha.copy()
        a_k[a_k != 1] = 0  # set all non foreground pixels to 0
        a_k[a_k == 1] = 1  # set all foreground pixels to 1

    print('intra u information flow.')
    # w_uu = intra_u(alpha, params['k_uu'], feature_uu)
    w_uu = uu(image, trimap, params['k_uu'], feature_uu)

    print('local information flow.')
    w_l = local(image, trimap)

    known = alpha.copy()
    known[(alpha == 1) | (alpha == 0)] = 1
    known[(alpha != 1) & (alpha != 0)] = 0
    T = diags(known, format='csr')  # CSR 存稀疏矩阵的一种方法

    solution = eq1(w_cm, w_uu, w_l, n_p, T, a_k, w_f, params)
    alpha = np.minimum(np.maximum(solution.reshape(image.shape[0], image.shape[1]), 0), 1)
    return alpha


if __name__ == '__main__':

    # file url
    input_dir = './data/input_lowres/'
    tri_map_dir = './data/trimap_lowres/Trimap1/'
    save_dir = './out/test/'
    file_name = 'plasticbag.png'
    input_file = input_dir + file_name
    tri_map_file = tri_map_dir + file_name
    save_file = save_dir + file_name

    # load image
    input_image = plt.imread(input_file)  # (x, y, 3)
    tri_map = plt.imread(tri_map_file)  # 0.50196081 Grey
    tri_map = tri_map[:, :, 0]  # Grey R=G=B, only use R. (x, y, 1)

    # mkdir and touch
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_file):
        os.system(r"touch {}".format(save_file))

    # matting
    alpha_matte = matting(input_image, tri_map)

    # save
    plt.imsave(save_file, alpha_matte, cmap='Greys_r')

    # show
    plt.imshow(alpha_matte, cmap='Greys_r')
    plt.show()
