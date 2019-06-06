import math

import numpy as np
from scipy.sparse import csr_matrix

from IFM.find_non_local_neighbors import find_non_local_neighbors


def intra_u(image, trimap, k, features):

    h, w, c = image.shape
    n = h * w
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    if is_unknown is None:
        is_unknown = np.ones((h, w), dtype=np.bool8)

    _, neigh_ind = find_non_local_neighbors(image, k, features, is_unknown, is_unknown)

    # This behaviour below, decreasing the xy-weight and finding a new set of neighbors, is taken
    # from the public implementation of KNN matting by Chen et al.
    features[:, -2:] /= 100.0
    in_ind, neigh_ind2 = find_non_local_neighbors(image, math.ceil(k / 5), features, is_unknown, is_unknown)

    neigh_ind = np.concatenate([neigh_ind, neigh_ind2], axis=1)

    in_ind = np.repeat(in_ind, neigh_ind.shape[1]).reshape(-1, neigh_ind.shape[1])
    flows = 1 - np.mean(np.abs(features[in_ind] - features[neigh_ind]), axis=2)
    flows[flows < 0] = 0

    w_uu = csr_matrix((flows.flatten(), (in_ind.flatten(), neigh_ind.flatten())), shape=(n, n))

    w_uu = 0.5 * (w_uu + w_uu.T)

    return w_uu
