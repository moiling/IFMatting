#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 02:30
# @Author  : moiling
# @File    : local_linear_embedding.py
import numpy as np

"""
% Local Linear Embedding
% This function implements the weight computation defined in
% Sam T. Roweis, Lawrence K. Saul, "Nonlinear Dimensionality
% Reduction by Local Linear Embedding", Science, 2000.
% 'w' is the weights for representing the row-vector 'pt' in terms
% of the dimensions x neighborCount matrix 'neighbors'.
% 'conditionerMult' is the multiplier of the identity matrix added
% to the neighborhood correlation matrix before inversion.
"""


def local_linear_embedding(pt, neighbors, conditioner_mult):
    # each column of neighbors represent a neighbor, each row a dimension
    # pt is a row vector
    # neighbors (n, 5)
    # pt (5, 1)
    # corr (n, 5) * (5, n) + (n, n) = (n, n)
    # pt_dot_n = (n, 5) * (5, 1) = (n, 1)
    corr = neighbors.dot(neighbors.T) + np.eye(neighbors.shape[0]) * conditioner_mult

    pt_dot_n = neighbors.dot(pt)

    alpha = 1 - np.sum(np.linalg.lstsq(corr, pt_dot_n, rcond=None)[0])  # 1 - sum(corr \ pt_dot_n)
    beta = np.sum(np.linalg.lstsq(corr, np.ones((corr.shape[0], 1)), rcond=None)[0])  # sum of elements of inv(corr)
    lagrange_mult = alpha / beta
    w = np.linalg.lstsq(corr, (pt_dot_n + lagrange_mult), rcond=None)[0]
    return w
