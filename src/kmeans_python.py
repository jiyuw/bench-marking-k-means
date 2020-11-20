import numpy as np


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
    :param X: points coordinate (n, 2)
    :param centroid: centroid coordinate (k, 2)
    :return: euclidean distance
    """
    n_samples = X.shape[0]
    n_clusters = centroid.shape[0]

    pts_x_mat = np.array([X[:, 0], ] * n_clusters).T
    pts_y_mat = np.array([X[:, 1], ] * n_clusters).T

    ctd_x_mat = np.array([centroid[:,0],] * n_samples)
    ctd_y_mat = np.array([centroid[:,1],] * n_samples)

    return (pts_x_mat-ctd_x_mat)**2+(pts_y_mat-ctd_y_mat)**2

def kmeans_py(X, cen, max_iter):
    """
    k means with python
    :param X: input data - (n,2)
    :param cen: initial centroids - (k,2)
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

    niter = 0
    while niter < max_iter:
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

        niter += 1

    return data_group, centroids_output, cost_output