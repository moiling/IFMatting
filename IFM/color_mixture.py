import numpy as np
from scipy.sparse import csr_matrix

from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.utils import solve_for_weights, local_linear_embedding


def color_mixture(image, trimap, k, features, use_xy_in_lle=False):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    h, w, c = image.shape
    n = h * w

    out_map = np.ones((h, w), dtype=np.bool8)

    in_ind, neigh_ind = find_non_local_neighbors(image, k, features, is_unknown, out_map)

    if not use_xy_in_lle:
        features = features[:, :-2]

    in_ind = np.repeat(in_ind, k).reshape(-1, k)

    flows = solve_for_weights(features[in_ind] - features[neigh_ind], regularization_factor=1e-10)

    w_cm = csr_matrix((flows.flatten(), (in_ind.flatten(), neigh_ind.flatten())), shape=(n, n))
    return w_cm
