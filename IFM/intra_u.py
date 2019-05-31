#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 17:23
# @Author  : moiling
# @File    : intra_u.py
import math

import numpy as np
from scipy.sparse import csr_matrix

from IFM.find_non_local_neighbors import find_non_local_neighbors

"""
% Color Similarity Non-local Pixel Affinities
% This function implements the affinity based on color differences 
% first used for image matting in the paper
% Qifeng Chen, Dingzeyu Li, Chi-Keung Tang, "KNN Matting", IEEE 
% TPAMI, 2013.
% All parameters other than image are optional. The output is a sparse
% matrix which has non-zero element for the non-local neighbors of
% the pixels given by binary map inMap.
% - K defines the number of neighbors from which LLE weights are 
%   computed.
% - outMap is a binary map that defines where the nearest neighbor 
%   search is done.
% - xyWeight determines how much importance is given to the spatial
%   coordinates in the nearest neighbor selection.
% - When useHSV is false (default), the search is done i [r g b x y] space,
%   otherwise the feature space is [cos(h) sin(h), s, v, x, y].
"""


def intra_u_2(image, trimap, k, features):
    h, w, _ = image.shape
    n = h * w

    # MATLAB: need_search_map = trimap != 1 & trimap != 0
    need_search_map = trimap.copy()
    need_search_map[(trimap != 1) & (trimap != 0)] = 1
    need_search_map[(trimap == 1) | (trimap == 0)] = 0

    search_map = need_search_map.copy()

    _, neighbors_indices = find_non_local_neighbors(image, k, features, need_search_map, search_map)

    # This behaviour below, decreasing the xy-weight and finding a new set of neighbors, is taken
    # from the public implementation of KNN matting by Chen et al.

    features[:, -2:] = features[:, -2:] / 100
    in_indices, neighbors_indices_2 = find_non_local_neighbors(image, math.ceil(k / 5), features, need_search_map, search_map)
    neighbors_indices = np.hstack((neighbors_indices, neighbors_indices_2))

    # different. author: / 100; why?
    features[:, -2:] = features[:, -2:] * 100

    in_indices = np.tile(in_indices.reshape(in_indices.shape[0], 1), neighbors_indices.shape[1])

    # MATLAB:max(1 - sum(abs(features(in_indices(:), :) - features(neighbors_indices(:), :)), 2) / size(features, 2), 0)
    # (10)
    flows = np.max(1 - np.sum(np.abs(features[in_indices.flatten(), :] - features[neighbors_indices.flatten(), :]), 1)
                   .reshape(in_indices.flatten().shape[0], 1) / features.shape[1], axis=1, initial=0)

    w_uu = csr_matrix((flows.flatten(), (in_indices.flatten(), neighbors_indices.flatten())), shape=(n, n))

    w_uu = (w_uu + w_uu.T) / 2
    return w_uu


def intra_u(image, trimap, k, features):

    h, w, c = image.shape
    N = h * w
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    if is_unknown is None:
        is_unknown = np.ones((h, w), dtype=np.bool8)

    _, neighInd = find_non_local_neighbors(image, k, features, is_unknown, is_unknown)

    # This behaviour below, decreasing the xy-weight and finding a new set of neighbors, is taken
    # from the public implementation of KNN matting by Chen et al.
    inInd, neighInd2 = find_non_local_neighbors(image, int(np.ceil(k / 5)), features[:, -2:] / 100, is_unknown, is_unknown)

    neighInd = np.concatenate([neighInd, neighInd2], axis=1)
    features[:, -2:] /= 100.0

    inInd = np.repeat(inInd, neighInd.shape[1]).reshape(-1, neighInd.shape[1])
    flows = 1 - np.mean(np.abs(features[inInd] - features[neighInd]), axis=2)
    flows[flows < 0] = 0

    W = csr_matrix((flows.flatten(), (inInd.flatten(), neighInd.flatten())), shape=(N, N))

    W = 0.5 * (W + W.T)

    return W