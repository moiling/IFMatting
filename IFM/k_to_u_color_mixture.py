import numpy as np
from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.utils import solve_for_weights


def k_to_u(image, trimap, k, features):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    # Find neighbors of unknown pixels in FG and BG
    in_ind, bg_ind = find_non_local_neighbors(image, k, features, is_unknown, is_bg)
    _, fg_ind = find_non_local_neighbors(image, k, features, is_unknown, is_fg)

    neigh_ind = np.concatenate([fg_ind, bg_ind], axis=1)

    # Compute LLE weights and estimate FG and BG colors that got into the mixture
    features = features[:, :-2]

    flows = solve_for_weights(features[in_ind].reshape(-1, 1, 3) - features[neigh_ind], 1e-10)

    fg_cols = np.sum(features[neigh_ind[:, :k]] * flows[:, :k, np.newaxis], axis=1)
    bg_cols = np.sum(features[neigh_ind[:, k:]] * flows[:, k:, np.newaxis], axis=1)

    alpha_est = trimap.copy()
    alpha_est[is_unknown] = np.sum(flows[:, :k], axis=1)

    # Compute the confidence based on FG - BG color difference
    confidence_of_unknown = np.sum(np.square(fg_cols - bg_cols), 1) / 3

    conf = is_known.astype(np.float64)
    conf[is_unknown] = confidence_of_unknown

    return alpha_est, conf
