from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
import sys
import numpy as np


def load_dataset_sp(sc, file):
    """
    load dataset
    :param sc: spark session
    :param file: path to file
    :return: RDD of data points
    """
    data = sc.textFile(file).map(lambda row:row.strip().split()).map(lambda row: (int(row[0]), int(row[1])))
    return data


def euc_dist_sp(point, centroid):
    """
    calculate euclidean distance between a data point and a centroid
    :param X: data point
    :param centroid:
    :return:
    """
    return (point[0]-centroid[0])**2+(point[1]-centroid[1])**2

def find_cen_idx(centroid, target):
    """
    find index of target in centroid list
    :param centroid: array of centroids
    :param target: target to find
    :return: index of target in centroid
    """
    for idx in range(centroid.shape[0]):
        if target == centroid[idx,:]:
            return idx


def kmeans_sp(X, cen, max_iter):
    """
    k means with spark
    :param X: RDD, input data
    :param cen: RDD, initial centroids
    :param max_iter: max iteration number
    :return: data_group - list of assigned clusters in each iteration
            centroids_output - list of centroids in each iteration
            cost_output - list of cost in each iteration
    """
    # output - not output intermediate centroids and label assignments
    cost_output = []

    niter = 0
    while niter < max_iter:
        # create data point - centroid pair, ((data_1, data_2), (cen_1, cen_2))
        X_cen_pair = X.cartesian(cen)
        if niter == max_iter-1:
            final_cen = np.array(cen.collect())

        # calculate distance between all points to all centroids
        X_cen_pair = X_cen_pair.map(lambda pair: (pair[0], (pair[1], euc_dist_sp(pair[0], pair[1]))))

        # find closest centroid of all points
        closest_cen = X_cen_pair.reduceByKey(lambda a, b: a if a[1]<b[1] else b)

        # calculate cost
        cost = closest_cen.map(lambda pair: pair[1][1]).sum()
        cost_output.append(cost)

        if niter == max_iter-1:
            final_assign = closest_cen.map(lambda pair:(pair[0], find_cen_idx(final_cen, pair[1][0])))
            final_assign = np.array(final_assign.collect())
            break

        # change format to use centroid as keys
        cen_X_pair = closest_cen.map(lambda pair: (pair[1][0], (pair[0], 1)))

        # re-calculate centroids
        cen = cen_X_pair.reduceByKey(lambda a, b: ((a[0][0]+b[0][0], a[0][1]+b[0][1]),a[1]+b[1]))\
                        .map(lambda pair: (pair[1][0][0]/pair[1][1], pair[1][0][1]/pair[1][1]))

        niter += 1

    return final_cen, final_assign, cost_output