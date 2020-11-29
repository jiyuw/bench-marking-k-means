import numpy as np
from tqdm.notebook import trange

def load_dataset_py(file):
    """
    load txt file as numpy array
    :param file: path to file
    :return: np.array
    """
    return np.loadtxt(file)


def euc_dist_py(X, centroid):
    """
    calculate euclidean distance
    :param X: points coordinate (n_samples, n_features)
    :param centroid: centroid coordinate (n_clusters, n_features)
    :return: euclidean distance
    """
    n_samples, n_features = X.shape
    n_clusters = centroid.shape[0]

    dist = np.zeros(shape=(n_samples, n_clusters))
    for i in range(n_features):
        pts_mat = np.array([X[:, i], ] * n_clusters).T
        ctd_mat = np.array([centroid[:,i],] * n_samples)
        dist += (pts_mat-ctd_mat)**2

    return dist

def kmeans_py(X, cen, max_iter):
    """
    k means with python
    :param X: input data - (n_samples, n_features)
    :param cen: initial centroids - (n_clusters, n_features)
    :param max_iter: max iteration number
    :return: data_group - list of assigned clusters in each iteration
            centroids_output - list of centroids in each iteration
            cost_output - list of cost in each iteration
    """
    n_clusters = cen.shape[0]

    # output
    data_group = []
    centroids_output = []
    cost_output = []

    for i in trange(max_iter):
        centroids_output.append(cen)

        # calculate enclidean distance between datapoints and centroids
        dist_mat = euc_dist_py(X, cen)

        # determine closest centroid for each point
        data_close_cen = np.argmin(dist_mat, axis=1)
        data_group.append(data_close_cen)

        # calculate cost and assign new centroid
        total_cost = 0
        for i in range(n_clusters):
            curr_cen = cen[i]
            curr_data = X[data_close_cen == i]
            num_data = curr_data.shape[0]
            tmp_cen = np.array([curr_cen,]*num_data)
            cost = np.sum((curr_data-tmp_cen)**2)
            total_cost += cost

            # create new centroid
            cen[i] = np.mean(curr_data, axis=0)
        cost_output.append(total_cost)

    return np.array(data_group), np.array(centroids_output), np.array(cost_output)