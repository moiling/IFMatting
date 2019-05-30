#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 02:18
# @Author  : moiling
# @File    : color_mixture.py
import numpy as np
from scipy.sparse import csr_matrix

from IFM.find_non_local_neighbors import find_non_local_neighbors
from IFM.local_linear_embedding import local_linear_embedding

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
    h, w, _ = image.shape
    n = h * w

    # MATLAB: need_search_map = trimap != 1 & trimap != 0
    need_search_map = trimap.copy()
    need_search_map[(trimap != 1) & (trimap != 0)] = 1
    need_search_map[(trimap == 1) | (trimap == 0)] = 0

    search_map = np.ones((h, w))

    in_indices, neighbors_indices = find_non_local_neighbors(image, k, features, need_search_map, search_map)

    flows = np.zeros((in_indices.shape[0], neighbors_indices.shape[1]))

    # use lle to solve (1)
    for i in range(in_indices.shape[0]):
        flows[i, :] = local_linear_embedding(features[in_indices[i], :].T,
                                             features[neighbors_indices[i, :].T, :],
                                             1e-10)

    flows = flows / np.tile(np.sum(flows, 1).reshape(flows.shape[0], 1), k)
    in_indices = np.tile(in_indices.reshape(in_indices.shape[0], 1), k)

    w_cm = csr_matrix((flows.flatten(), (in_indices.flatten(), neighbors_indices.flatten())), shape=(n, n))

    return w_cm
