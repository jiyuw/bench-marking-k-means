"""
script used in EMR cluster to run
submit as spark application
run '/bin/spark-submit 003-kmeans_spark.py <dataset> <mode>' in master node
"""

import sys
from pyspark import SparkContext
from pyspark.sql import *
import os
import pandas as pd
import numpy as np
import time
from tqdm import trange

def load_dataset_sp(sc, file):
    """
    load dataset
    :param sc: spark session
    :param file: path to file
    :return: RDD of data points
    """
    data = sc.textFile(file).map(lambda row:row.strip().split()).map(lambda row: (int(row[0]), int(row[1])))
    return data

def euc_dist_sp(point, centroids):
    """
    calculate euclidean distance between a data point and a centroid
    :param X: data point
    :param centroids: list of centroids
    :return:
    """
    point = np.array(point)
    min_dist = float('inf')
    for i in range(len(centroids)):
        dist = np.sum((point-centroids[i])**2)
        if dist < min_dist:
            idx = i
    return idx

def flatten_tuple(t):
    """
    flatten tuple output
    :param t: (cen_idx, (point, 1))
    :return:
    """
    return tuple(list(t[1][0])+[t[0]])

def kmeans_sp(con, X, cen, max_iter):
    """
    k means with spark
    :param con: spark session
    :param X: RDD, input data
    :param cen: RDD, initial centroids
    :param max_iter: max iteration number
    :return: data_group - list of assigned clusters in each iteration
            centroids_output - list of centroids in each iteration
            cost_output - list of cost in each iteration
    """
    for i in trange(max_iter):
        # collect centroids
        centroids = np.array(cen.collect())
        if i == max_iter - 1:
            final_cen = centroids

        # calculate distance between all points to all centroids and find closest one: (cen_idx, (point, 1))
        closest_cen = X.map(lambda pair: (euc_dist_sp(pair, centroids), (np.array(pair), 1)))

        if i == max_iter-1:
            final_assign = closest_cen.map(lambda pair: flatten_tuple(pair)).collect()
            break

        # re-calculate centroids
        cen = closest_cen.reduceByKey(lambda a, b: (a[0]+b[0],a[1]+b[1]))\
                        .map(lambda pair: tuple(pair[1][0]/pair[1][1]))

    return np.array(final_assign), np.array(final_cen), None

def millitime(t):
    return round(t*1000, 3)

class dataset():
    def __init__(self, name, location, con=None, output_dir="results/python"):
        if location not in ['python', 'spark', 'snowflake']:
            raise ValueError("wrong type")
        if location in ['spark', 'snowflake'] and not con:
            raise ValueError("need connector for spark or snowflake")
        self.name = name
        self.con = con
        self.location = location
        self.output_file = {'label': os.path.join(output_dir, self.name + f"-label_{self.location}.txt"),
                            'centroid': os.path.join(output_dir, self.name + f"-cen_{self.location}.txt"),
                            'cost': os.path.join(output_dir, self.name + f'-cost_{self.location}.txt')}
        print("#### "+name)
        print(f"- {name} initiated")

    def load_dataset(self, load_fn, root_dir="datasets"):
        if self.con:
            self.data = load_fn(self.con, os.path.join(root_dir, self.name + '.txt'))
            self.gt_par = load_fn(self.con, os.path.join(root_dir, self.name + '-pa.txt'))
            self.gt_cen = load_fn(self.con, os.path.join(root_dir, self.name + '-c.txt'))
            self.init_cen = load_fn(self.con, os.path.join(root_dir, self.name + '-ic.txt'))
        else:
            self.data = load_fn(os.path.join(root_dir, self.name + '.txt'))
            self.gt_par = load_fn(os.path.join(root_dir, self.name + '-pa.txt'))
            self.gt_cen = load_fn(os.path.join(root_dir, self.name + '-c.txt'))
            self.init_cen = load_fn(os.path.join(root_dir, self.name + '-ic.txt'))
        print(f"- {self.name} data loaded")

    def dataset_info(self):
        n_sample, n_feature = self.data.shape
        n_cluster = self.gt_cen.shape[0]
        return n_sample, n_feature, n_cluster

    def kmeans_train(self, train_fn, max_iter):
        print(f"- {self.name} training start")
        start = time.time()
        if self.location == 'python':
            self.label, self.centroid, self.cost = train_fn(self.data, self.init_cen, max_iter)
        else:
            self.label, self.centroid, self.cost = train_fn(self.con, self.data, self.init_cen, max_iter)
        end = time.time()
        print(f"- {self.name} trained")
        t = millitime(end-start)
        print(f"time used: {t}ms")
        return t

    def save_output(self):
        np.savetxt(self.output_file['label'], self.label)
        if self.cost:
            np.savetxt(self.output_file['cost'], self.cost)
        if self.location == 'python':
            np.savetxt(self.output_file['centroid'], self.centroid.reshape(self.centroid.shape[0], -1), fmt='%d')
        else:
            np.savetxt(self.output_file['centroid'], self.centroid)
        print(f"- {self.name} saved")

    def load_output(self):
        self.label = np.loadtxt(self.output_file['label'])
        self.cost = np.loadtxt(self.output_file['cost'])
        self.centroid = np.loadtxt(self.output_file['centroid'])
        if self.location == 'python':
            self.centroid = self.centroid.reshape((self.centroid.shape[0], -1, 2))
        print(f"- {self.name} output loaded")

    def eval_output(self):
        if self.cost:
            print(f"The final cost reduction is {round((self.cost[-2]-self.cost[-1])/self.cost[-2]*100, 2)}%")

def main(name, sc, mode):
    cols = ['dataset', 'time']
    if not os.path.exists(f'results/time_spark-{mode}.csv'):
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.read_csv(f'results/time_spark-{mode}.csv')
    names = [name, name+'e1', name+'e2', name+'e3']
    for n in names:
        if n in list(df['dataset']):
            print("#### "+n+" skipped")
            continue
        d = dataset(n, 'spark', sc, 'results')
        d.load_dataset(load_dataset_sp, 's3://data516project/datasets')
        t = d.kmeans_train(kmeans_sp, 100)
        d.save_output()
        tmp = pd.DataFrame([[n, t]], columns=cols)
        df = df.append(tmp, ignore_index=True)
        del d
        df.to_csv(f'results/time_spark-{mode}.csv', index=False)

def test(name, sc):
    cols = ['dataset', 'time']
    if not os.path.exists('results/time_spark.csv'):
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.read_csv('results/time_spark.csv')
    n = name
    d = dataset(n, 'spark', sc, 'results')
    d.load_dataset(load_dataset_sp, 's3://data516project/datasets')
    t = d.kmeans_train(kmeans_sp, 2)
    d.save_output()
    tmp = pd.DataFrame([[n, t]], columns=cols)
    df = df.append(tmp, ignore_index=True)
    del d
    print(df)

if __name__ == '__main__':
    name = sys.argv[1]
    mode = sys.argv[2]

    sc = SparkContext()
    spark = SparkSession.builder.getOrCreate()

    if mode == 'test':
        test(name, sc)
    else:
        time_df = main(name, sc, mode)
    sc.stop()