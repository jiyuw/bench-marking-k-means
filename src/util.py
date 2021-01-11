import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.metrics import normalized_mutual_info_score as NMI
import os
import time

def cen_init(X, n_clusters, random_state, n_local_trials=None):
    """
    Init n_clusters seeds according to k-means++
    Modified from: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/cluster/_kmeans.py#L749
    :param X: ndarray of shape (n_samples, n_features)
    :param n_clusters: int, the number of seeds to choose
    :param random_state: RandomState instance. The generator used to initialize the centers.
    :param n_local_trials: int, default=None. The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    :return:
    """
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    x_squared_norms = row_norms(X, squared=True)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]

    return centers

def expand_dataset(X, cen, pa, target_n):
    n_samples, n_features = X.shape
    n_clusters = cen.shape[0]

    new_sample_per_cluster = (target_n-n_samples)//n_clusters

    for i in range(n_clusters):
        curr_cluster = np.array([i+1]*new_sample_per_cluster)
        curr_cen = cen[i, :]
        tmp = np.zeros((new_sample_per_cluster, n_features))
        for j in range(n_features):
            tmp[:, j] = np.random.normal(curr_cen[j], 1, size=new_sample_per_cluster).astype(int)
        X = np.append(X, tmp, axis=0)
        pa = np.append(pa, curr_cluster)

    return X, pa


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
        if self.cost:
            self.cost = np.loadtxt(self.output_file['cost'])
        self.centroid = np.loadtxt(self.output_file['centroid'])
        if self.location == 'python':
            self.centroid = self.centroid.reshape((self.centroid.shape[0], -1, 2))
        print(f"- {self.name} output loaded")

    def eval_output(self):
        if self.cost:
            print(f"The final cost reduction is {round((self.cost[-2]-self.cost[-1])/self.cost[-2]*100, 2)}%")
        if self.location == 'python':
            final_assign = self.label[-1,:]
        else:
            final_assign = self.label
        nmi = NMI(self.gt_par, final_assign)
        print(f"The final NMI score is {round(nmi, 4)}")
        return nmi