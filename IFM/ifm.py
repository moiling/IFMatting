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
import inspect

import numpy as np
import scipy
from sklearn.neighbors import KDTree

from scipy.sparse import csr_matrix as csr, diags

from IFM.lle import barycenter_kneighbors_graph_ku as bkg_ku
from IFM.color_mixture import color_mixture as cm
from IFM.intra_u import intra_u as uu
from IFM.local import local
import matplotlib.pyplot as plt


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


def solve_alpha(w_cm, w_uu, w_l, n_p, tau, a_k, w_f, params):
    d_cm = diags(csr.sum(w_cm, axis=1).A.ravel(), format='csr')
    d_uu = diags(csr.sum(w_uu, axis=1).A.ravel(), format='csr')
    d_l = diags(csr.sum(w_l, axis=1).A.ravel(), format='csr')

    # (15)
    l_ifm = (d_cm - w_cm).T.dot(d_cm - w_cm) + params['s_uu'] * (d_uu - w_uu) + params['s_l'] * (d_l - w_l)

    if params['use_k_u']:
        # (16)
        A = l_ifm + params['lambda'] * tau + params['s_ku'] * n_p
        b = (params['lambda'] * tau + params['s_ku'] * n_p).dot(w_f)
    else:
        # (19)
        A = l_ifm + params['lambda'] * tau
        b = params['lambda'] * tau * a_k

    # use preconditioned conjugate gradient to solve the linear systems
    solution = scipy.sparse.linalg.cg(A, b, x0=w_f, tol=1e-10, maxiter=5000, M=None, callback=report)
    return solution[0]


def report(x):
    frame = inspect.currentframe().f_back
    print('%4d: %e' % (frame.f_locals['iter_'], frame.f_locals['resid']))


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


def information_flow_matting(image, trimap, use_k_u=False):
    # load image
    image = plt.imread(image)  # (x, y, 3)
    trimap = plt.imread(trimap)  # 0.50196081 Grey
    trimap = trimap[:, :, 0]  # Grey R=G=B, only use R. (x, y, 1)

    print('Start matting.')

    h, w, _ = image.shape
    params, feature_cm, feature_ku, feature_uu = init_params(image, h, w)
    alpha = trimap.flatten()

    print('Color-mixture information flow.')
    w_cm = cm(image, trimap, params['k_cm'], feature_cm)

    # TODO 判断是否执行k_u
    params['use_k_u'] = use_k_u

    if params['use_k_u']:
        print('K-to-U information flow.')
        w_f, n_p = k_to_u(alpha, params['k_ku'], feature_ku)
        a_k = None
    else:
        w_f = None
        n_p = None

        # αK is a row vector with pth entry being 1 if p ∈ F and 0 otherwise
        a_k = alpha.copy()
        a_k[a_k != 1] = 0  # set all non foreground pixels to 0
        a_k[a_k == 1] = 1  # set all foreground pixels to 1

    print('intra u information flow.')
    w_uu = uu(image, trimap, params['k_uu'], feature_uu)

    print('local information flow.')
    w_l = local(image, trimap)

    known = alpha.copy()
    known[(alpha == 1) | (alpha == 0)] = 1
    known[(alpha != 1) & (alpha != 0)] = 0

    # Tau is an N × N diagonal matrix with diagonal entry (p, p) 1 if p ∈ K and 0 otherwise
    tau = diags(known, format='csr')

    print('solve alpha.')
    solution = solve_alpha(w_cm, w_uu, w_l, n_p, tau, a_k, w_f, params)
    alpha = np.minimum(np.maximum(solution.reshape(image.shape[0], image.shape[1]), 0), 1)
    return alpha
