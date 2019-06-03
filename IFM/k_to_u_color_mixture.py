import numpy as np
from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.utils import solve_for_weights


def k_to_u(image, trimap, k, features):
    n = features.shape[0]
    index = np.arange(n)

    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    unknown_index = index[is_unknown.flatten()]

    # Find neighbors of unknown pixels in FG and BG
    inInd, bgInd = find_non_local_neighbors(image, k, features, is_unknown, is_bg)
    _, fgInd = find_non_local_neighbors(image, k, features, is_unknown, is_fg)

    neighInd = np.concatenate([fgInd, bgInd], axis=1)

    # Compute LLE weights and estimate FG and BG colors that got into the mixture
    features = features[:, :-2]
    flows = np.zeros((inInd.shape[0], neighInd.shape[1]))
    fgCols = np.zeros((inInd.shape[0], 3))
    bgCols = np.zeros((inInd.shape[0], 3))

    flows = solve_for_weights(features[inInd].reshape(-1, 1, 3) - features[neighInd], 1e-10)

    fgCols = np.sum(features[neighInd[:, :k]] * flows[:, :k, np.newaxis], axis=1)
    bgCols = np.sum(features[neighInd[:, k:]] * flows[:, k:, np.newaxis], axis=1)

    alphaEst = trimap.copy()
    alphaEst[is_unknown] = np.sum(flows[:, :k], axis=1)

    # Compute the confidence based on FG - BG color difference
    confidence_of_unknown = np.sum(np.square(fgCols - bgCols), 1) / 3

    conf = is_known.astype(np.float64)
    conf[is_unknown] = confidence_of_unknown

    return alphaEst, conf