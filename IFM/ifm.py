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

import numpy as np

from scipy.sparse import diags

from IFM.color_mixture import color_mixture as cm
from IFM.intra_u import intra_u as uu
from IFM.k_to_u_color_mixture import k_to_u
from IFM.local import local
import matplotlib.pyplot as plt

from IFM.solve_alpha import solve_alpha
from utils.utils import patchBasedTrimming


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
        's_cm': 1,
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
        patchTrimmed = patchBasedTrimming(image, trimap, 0.25, 0.9, 1, 5)

        print('K-to-U information flow.')
        # w_f, n_p = k_to_u(alpha, params['k_ku'], feature_ku)
        kToU, kToUconf = k_to_u(image, patchTrimmed, params['k_ku'], feature_ku)
        a_k = None
    else:
        kToUconf = None
        kToU = None
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

    print('Solving for alphas.')
    # solution = solve_alpha(w_cm, w_uu, w_l, n_p, tau, a_k, w_f, params)
    solution = solve_alpha(trimap, w_cm, w_uu, w_l, kToUconf, tau, a_k, kToU, params)
    # alpha_matte = np.minimum(np.maximum(solution.reshape(image.shape[0], image.shape[1]), 0), 1)
    alpha_matte = np.clip(solution, 0, 1).reshape(trimap.shape)

    return alpha_matte
