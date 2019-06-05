import numpy as np
from sklearn.neighbors import KDTree


def find_non_local_neighbors(image, k, features, need_search_map, search_map, erase_self_matches=True):
    h, w, c = image.shape

    if need_search_map is None:
        need_search_map = np.ones((h, w), dtype=np.bool8)

    if search_map is None:
        search_map = np.ones((h, w), dtype=np.bool8)

    if features is None:
        features = np.stack([
            image[:, :, 0].flatten(),
            image[:, :, 1].flatten(),
            image[:, :, 2].flatten(),
        ], axis=1)

    indices = np.arange(h * w)

    in_map = need_search_map.flatten()
    out_map = search_map.flatten()

    in_ind = indices[in_map]
    out_ind = indices[out_map]

    kd_tree = KDTree(features[out_map, :], leaf_size=30, metric='euclidean')

    if erase_self_matches:
        # Find K + 1 matches to count for self-matches
        neighbors = kd_tree.query(features[in_map, :], k=k + 1, return_distance=False)
        # Get rid of self-matches
        valid_neighbors_map = np.ones(neighbors.shape, dtype=np.bool)
        in_and_out_map = np.logical_and(in_map[in_ind], out_map[in_ind])

        # 如果区域重复，最近的一个点就是自己
        valid_neighbors_map[in_and_out_map, 0] = False
        valid_neighbors_map[:, -1] = ~valid_neighbors_map[:, 0]
        valid_neighbors = np.zeros((neighbors.shape[0], neighbors.shape[1] - 1), dtype=np.int)

        for i in range(valid_neighbors.shape[0]):
            valid_neighbors[i, :] = neighbors[i, valid_neighbors_map[i, :]]
        neighbors_indices = out_ind[valid_neighbors]
    else:

        neighbors = kd_tree.query(features[in_map, :], k=k, return_distance=False)
        neighbors_indices = out_ind[neighbors]

    return in_ind, neighbors_indices
