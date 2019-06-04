#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 02:18
# @Author  : moiling
# @File    : color_mixture.py
import numpy as np
from scipy.sparse import csr_matrix

from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.utils import solve_for_weights

"""
% Color Mixture Non-local Pixel Affinities
% This function implements the color-mixture information flow in
% Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "Designing Effective 
% Inter-Pixel Information Flow for Natural Image Matting", CVPR, 2017
% when the input parameter 'useXYinLLEcomp' is false (default), and
% the affinity definition used in
% Xiaowu Chen, Dongqing Zou, Qinping Zhao, Ping Tan, "Manifold 
% preserving edit propagation", ACM TOG, 2012
% when 'useXYinLLEcomp' is true.
% All parameters other than image are optional. The output is a sparse
% matrix which has non-zero element for the non-local neighbors of
% the pixels given by binary map inMap.
% - K defines the number of neighbors from which LLE weights are 
%   computed.
% - outMap is a binary map that defines where the nearest neighbor 
%   search is done.
% - xyWeight determines how much importance is given to the spatial
%   coordinates in the nearest neighbor selection.
"""


def color_mixture(image, trimap, k, features):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    h, w, c = image.shape
    N = h * w

    if is_unknown is None:
        is_unknown = np.ones((h, w), dtype=np.bool8)

    outMap = np.ones((h, w), dtype=np.bool8)

    inInd, neighInd = find_non_local_neighbors(image, k, features, is_unknown, outMap)

    inInd = np.repeat(inInd, k).reshape(-1, k)
    flows = solve_for_weights(features[inInd] - features[neighInd], regularization_factor=1e-10)

    i = inInd.flatten()
    j = neighInd.flatten()
    v = flows.flatten()

    W = csr_matrix((flows.flatten(), (inInd.flatten(), neighInd.flatten())), shape=(N, N))

    return W