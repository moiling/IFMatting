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