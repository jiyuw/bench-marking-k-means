import numpy as np
import pandas as pd
import os
import sys


def initialize_centroid(n, k, seed=111):
    """
    initialize centroids randomly
    :param n: number of nodes in dataset
    :param k: number of clusters
    :return: index of k centroids
    """
    return np.random.randint(n, size=k)


def euclidean_distance(points, centroid):
    """
    calculate euclidean distance
    :param points: points coordinate (n, 2)
    :param centroid: centroid coordinate (k, 2)
    :return: euclidean distance
    """
    n = points.shape[0]
    k = centroid.shape[0]

    pts_x_mat = np.array([points[:,0],]*k).T
    pts_y_mat = np.array([points[:,1],]*k).T

    ctd_x_mat = np.array([centroid[:,0],]*n)
    ctd_y_mat = np.array([centroid[:,1],]*n)

    return (pts_x_mat-ctd_x_mat)**2+(pts_y_mat-ctd_y_mat)**2

def kmeans_py(data, k, max_iter, seed=111):
    n = data.shape[0]

    # output
    data_group = []
    centroids_output = []
    cost_output = []

    # initialize centroids
    cen = data[initialize_centroid(n, k, seed=seed)]

    niter = 0
    while niter < max_iter:
        centroids_output.append(cen)

        # calculate enclidean distance between datapoints and centroids
        dist_mat = euclidean_distance(data, cen)

        # determine closest centroid for each point
        data_close_cen = np.argmin(dist_mat, axis=1)
        data_group.append(data_close_cen)

        # calculate cost and assign new centroid
        total_cost = 0
        for i in range(k):
            curr_cen = cen[i]
            curr_data = data[data_close_cen==i]
            num_data = curr_data.shape[0]
            tmp_cen = np.array([curr_cen,]*num_data)
            cost = np.sum((curr_data-tmp_cen)**2)
            total_cost += cost

            # create new centroid
            cen[i] = np.mean(curr_data, axis=0)
        cost_output.append(total_cost)

        niter += 1

    return data_group, centroids_output, cost_output