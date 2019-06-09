# The information flow alpha matting method implementation in this file is based on
# https://github.com/yaksoy/AffinityBasedMattingToolbox
# by Yağız Aksoy.
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
import cv2
import numpy as np

from IFM.color_mixture import color_mixture as cm
from IFM.intra_u import intra_u as uu
from IFM.k_to_u_color_mixture import k_to_u
from IFM.local import local
from IFM.solve_alpha import solve_alpha
from utils.patch_based_trimming import patch_based_trimming


def compute_features(img, xy_weight, x, y, w, h):
    features = np.stack((
        img[:, :, 0].flatten(),
        img[:, :, 1].flatten(),
        img[:, :, 2].flatten(),
        x.astype(np.float64) * xy_weight / w,
        y.astype(np.float64) * xy_weight / h
    ), axis=1)
    return features


def init_params(img):
    h, w, c = img.shape

    params = {
        'k_cm': 20,
        'k_ku': 7,
        'k_uu': 5,
        's_cm': 1,
        's_ku': 0.05,
        's_uu': 0.01,
        's_l': 1,
        'lambda': 100,
        'xyw_cm': 1,
        'xyw_ku': 10,
        'xyw_uu': 0.05,
        'use_k_u': True,
        'use_patch_trimmed': True
    }

    # ˜x and ˜y are the image coordinates normalized by image width and height
    x = np.arange(1, w + 1)
    y = np.arange(1, h + 1)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    feature_cm = compute_features(img, params['xyw_cm'], x, y, w, h)
    feature_ku = compute_features(img, params['xyw_ku'], x, y, w, h)
    feature_uu = compute_features(img, params['xyw_uu'], x, y, w, h)

    return params, feature_cm, feature_ku, feature_uu


def information_flow_matting(image, trimap, use_k_u=False):
    image = cv2.imread(image)
    trimap = cv2.imread(trimap)

    image = image / 255
    trimap = trimap[:, :, 0] / 255

    print('Start matting.')
    params, feature_cm, feature_ku, feature_uu = init_params(image)

    # TODO detect highly transparent
    params['use_k_u'] = use_k_u

    # TODO edge trimmed

    print('Color-mixture information flow.')
    w_cm = cm(image, trimap, params['k_cm'], feature_cm)

    if params['use_k_u']:
        print('K-to-U information flow.')
        if params['use_patch_trimmed']:
            print('\tuse patch trimmed')
            patch_trimmed = patch_based_trimming(image, trimap, 0.25, 0.9, 1, 5)
            w_f, h = k_to_u(image, patch_trimmed, params['k_ku'], feature_ku)
        else:
            w_f, h = k_to_u(image, trimap, params['k_ku'], feature_ku)

        a_k = None
    else:
        h = w_f = None

        # αK is a row vector with pth entry being 1 if p ∈ F and 0 otherwise
        a_k = trimap.flatten()
        a_k[a_k != 1] = 0  # set all non foreground pixels to 0
        a_k[a_k == 1] = 1  # set all foreground pixels to 1

    print('Intra-u information flow.')
    w_uu = uu(image, trimap, params['k_uu'], feature_uu)

    print('Local information flow.')
    w_l = local(image, trimap)

    alpha = trimap.flatten()
    known = alpha.copy()
    known[(alpha == 1) | (alpha == 0)] = 1
    known[(alpha != 1) & (alpha != 0)] = 0

    print('Solving for alphas.')
    solution = solve_alpha(trimap, w_cm, w_uu, w_l, h, a_k, w_f, params)

    alpha_matte = np.clip(solution, 0, 1).reshape(trimap.shape)

    return alpha_matte
