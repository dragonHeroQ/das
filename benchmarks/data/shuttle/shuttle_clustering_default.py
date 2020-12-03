import sys
sys.path.append('../../')
from automl.get_algorithm import *
from automl.Clustering.default_keys import get_default_clustering_keys

import time
import warnings

import sklearn.datasets
warnings.filterwarnings("ignore")
import numpy as np
from automl.performance_evaluation import *
from load_shuttle import load_shuttle
X, y = load_shuttle(return_X_y=True)


if __name__ == "__main__":
    keys= [ "AffinityPropagation",
            "AgglomerativeClustering",
            "Birch",
            "DBSCAN",
            # "GaussianMixture",
            "KMeans",
            "MeanShift",
            "MiniBatchKMeans",
            "SpectralClustering"]
    for i in keys:
        try:
            mod = get_algorithm_by_key(i)
            mod.fit(X)
            y_hat = mod.predict(X)
            print(i, "s score", eval_performance("adjusted_mutual_info_score", y_true=y, y_score=y_hat))
        except:
            print("error.........")

# error.........
# error.........
# error.........
# DBSCAN s score 2.63219842025e-15
# KMeans s score 0.204314482978
# MeanShift s score 0.333556827671
# MiniBatchKMeans s score 0.19387750494
# error.........
