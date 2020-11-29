from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
import sys
import numpy as np
from tqdm.notebook import trange


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
    point = np.array(point)
    centroid = np.array(centroid)
    return (point-centroid)**2

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

    for i in trange(max_iter):
        # create data point - centroid pair, ((data), (cen))
        X_cen_pair = X.cartesian(cen)
        if i == max_iter-1:
            final_cen = np.array(cen.collect())

        # calculate distance between all points to all centroids: (point, (centroid, distance))
        X_cen_pair = X_cen_pair.map(lambda pair: (pair[0], (pair[1], euc_dist_sp(pair[0], pair[1]))))

        # find closest centroid of all points
        closest_cen = X_cen_pair.reduceByKey(lambda a, b: a if a[1]<b[1] else b)

        # calculate cost
        cost = closest_cen.map(lambda pair: pair[1][1]).sum()
        cost_output.append(cost)

        if i == max_iter-1:
            final_assign = closest_cen.map(lambda pair:(pair[0], find_cen_idx(final_cen, pair[1][0])))
            final_assign = np.array(final_assign.collect())
            break

        # change format to use centroid as keys
        cen_X_pair = closest_cen.map(lambda pair: (pair[1][0], (pair[0], 1)))

        # re-calculate centroids
        cen = cen_X_pair.reduceByKey(lambda a, b: (tuple(np.array(a[0])+np.array(b[0])),a[1]+b[1]))\
                        .map(lambda pair: tuple(np.array(pair[1][0])/pair[1][1]))

    return final_assign, final_cen, np.array(cost_output)