import numpy as np
from sklearn.neighbors.kd_tree import KDTree

from IFM.find_non_local_neighbors import find_non_local_neighbors
from utils.lle import barycenter_kneighbors_graph_ku as bkg_ku
from scipy.sparse import csr_matrix as csr

from utils.utils import solve_for_weights


def k_to_u_2(alpha, k, feature):

    # pixel size
    n = feature.shape[0]

    index = np.arange(n)

    fore = feature[alpha == 1]
    fore_index = index[alpha == 1]

    back = feature[alpha == 0]
    back_index = index[alpha == 0]

    unknown = feature[(alpha != 0) & (alpha != 1)]
    unknown_index = index[(alpha != 0) & (alpha != 1)]

    # nearest foreground pixel to unknown
    kd_tree = KDTree(fore, leaf_size=30, metric='euclidean')
    # nf 就是一个(n, 7)矩阵，每行表示找出的7个临近点，n是未知点个数，所以nf、nb的n是相同的
    nf = kd_tree.query(unknown, k=k, return_distance=False)
    # fore_index是一个n维向量，把nf放进去，其实就是找每一个对应的index是什么，返回的还是(n, 7)矩阵
    nf_index = fore_index[nf]

    # nearest background pixel to unknown
    kd_tree = KDTree(back, leaf_size=30, metric='euclidean')
    nb = kd_tree.query(unknown, k=k, return_distance=False)
    nb_index = back_index[nb]

    # 把两个(n, 7)合并成(n, 14)
    nfb_index = np.concatenate((nf_index, nb_index), axis=1)

    # feature[:,:-2]是[x, 3]，x为像素总数，在[x, 3]里找下标对应[n, 14]，变成[n, 14, 3]
    nfb_color = feature[:, :-2][nfb_index]
    unknown_color = unknown[:, :-2]

    # compute (1)
    w_ku = bkg_ku(unknown_color, nfb_color, nfb_index, n_neighbors=2 * k)

    # 为了和颜色相乘，要加一维
    w_ku = w_ku.reshape((w_ku.shape[0], w_ku.shape[1], 1))

    nfb_weighted_color = w_ku * nfb_color

    # 前7个是前景的, 计算公式(6)
    cpf = np.sum(nfb_weighted_color[:, :k, :], axis=1) / np.sum(w_ku[:, :k], axis=1)
    cpb = np.sum(nfb_weighted_color[:, k:, :], axis=1) / np.sum(w_ku[:, k:], axis=1)

    w_f = np.zeros(n)
    w_f[unknown_index] = np.sum(w_ku[:, 0:k, :], axis=1)[:, 0]
    w_f[fore_index] = 1
    # 公式(7)
    n_p = np.sum((cpf - cpb) * (cpf - cpb), axis=1) / 3
    n_p = csr((n_p, (unknown_index, unknown_index)), shape=(n, n))
    return w_f, n_p


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