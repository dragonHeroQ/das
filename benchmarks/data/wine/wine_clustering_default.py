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

X, y = sklearn.datasets.load_wine(return_X_y=True)


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

# AffinityPropagation s score 0.254488405741
# AgglomerativeClustering s score 0.325122277803
# Birch s score 0.409652508731
# DBSCAN s score -6.01300701455e-16
# KMeans s score 0.25619119423
# MeanShift s score 0.424873226061
# MiniBatchKMeans s score 0.265264031984
# SpectralClustering s score 9.49654731228e-05