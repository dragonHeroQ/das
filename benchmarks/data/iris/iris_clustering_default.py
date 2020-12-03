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

X, y = sklearn.datasets.load_iris(return_X_y=True)


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

# AffinityPropagation s score 0.517173162328
# AgglomerativeClustering s score 0.576770712041
# Birch s score 0.670610539064
# DBSCAN s score 0.554365073775
# KMeans s score 0.49139405805
# MeanShift s score 0.550981412904
# MiniBatchKMeans s score 0.477561279515
# SpectralClustering s score 0.492840228602