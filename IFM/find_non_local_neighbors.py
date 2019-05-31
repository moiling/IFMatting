#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 01:11
# @Author  : moiling
# @File    : find_non_local_neighbors.py
import numpy as np
from sklearn.neighbors import KDTree

"""
% Find neighbors using the pixel colors and spatial coordinates
% - K is the number of neighbors to be found
% - Parameters other than K and image are optional.
% - xyWeight sets the relative importance of spatial coordinates
% - inMap and outMap are binary maps determining the query and
%   search regions
% - Self matches are detected and removed if eraseSelfMatches is true
% - inInd and neighInd give pixel indices of query pixels and their neighbors.
% - features is noOfPixels X dimensions matrix used in neighbor search.
"""


def find_non_local_neighbors_2(image, k, features, need_search_map, search_map):
    """
    :param image:
    :param k:
    :param features:
    :param need_search_map: 需要寻找邻居的区域
    :param search_map: 用来寻找邻居的区域 -> 如CM中要在全图范围找U中每个点的邻居，need_search_map=u,search_map=all
    :return:
    """
    h, w, _ = image.shape

    in_map = need_search_map.flatten()
    out_map = search_map.flatten()
    indices = np.arange(h * w).T
    in_indices = indices[in_map != 0]
    out_indices = indices[out_map != 0]

    kd_tree = KDTree(features[out_map != 0, :], leaf_size=30, metric='euclidean')

    # TODO self-matches
    neighbors = kd_tree.query(features[in_map != 0, :], k=k, return_distance=False)
    neighbors_indices = out_indices[neighbors]

    return in_indices, neighbors_indices


def find_non_local_neighbors(image, k, features, need_search_map, search_map):
    h, w, c = image.shape

    if need_search_map is None:
        need_search_map = np.ones((h, w), dtype=np.bool8)

    if search_map is None:
        search_map = np.ones((h, w), dtype=np.bool8)


    indices = np.arange(h * w)

    inMap = need_search_map.flatten()
    outMap = search_map.flatten()

    inInd = indices[inMap]
    outInd = indices[outMap]

    if features is None:
        features = np.stack([
            image[:, :, 0].flatten(),
            image[:, :, 1].flatten(),
            image[:, :, 2].flatten(),
        ], axis=1)

    kd_tree = KDTree(features[outMap != 0, :], leaf_size=30, metric='euclidean')

    neighbors = kd_tree.query(features[inMap != 0, :], k=k, return_distance=False)
    neighbors_indices = outInd[neighbors]

    return inInd, neighbors_indices