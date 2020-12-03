import sys
sys.path.append('../../')
import time
import warnings
from load_shuttle import load_shuttle
from sklearn.model_selection import train_test_split
import sklearn.datasets
warnings.filterwarnings("ignore")
import numpy as np

X, y = load_shuttle(return_X_y=True)

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#print(len(x_train), len(x_test))

#from automl.Classification.classifier import Classifier
from automl.Clustering.cluster import Cluster
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
import time
from automl.performance_evaluation import eval_performance

import os

import warnings
warnings.filterwarnings("ignore")


validates_records = []
test_records_s = []
test_records_adj = []
time_cost_records = []
xx = [(i+1) * 60 for i in range(60)]
xxx = []

from sklearn.externals import joblib
for i in range(1):

    start_time = time.time()

    clf = Cluster(total_timebudget=360, per_run_timebudget=3600)
    clf.set_version(i)
    clf.configure_default_algorithm_set()

    clf.set_automl_mode(2)
    #clf.set_classification_mode(1)
    # clf._fit_based_on_q_learning(, y)
    clf.fit(X)

    y_hat = clf.predict(X)
    #y_hat = np.array(y_hat.tolist())
    #print(y_hat.shape)


    time_cost = time.time() - start_time
    time_cost_records.append(time_cost)
    validates_records.append(clf.get_best_val_score())



    tmp_res = eval_performance("silhouette_score", X=X, y_score=y_hat)

    test_records_s.append(tmp_res)

    tmp_res = eval_performance("adjusted_mutual_info_score", y_true=y, y_score=y_hat)

    test_records_adj.append(tmp_res)

    xxx.append(clf.get_best_model_at_specify_timepointset(xx))
    print("automl_mode_2 on iris clustering: ", tmp_res)
    print("Time_Cost: {}".format(time_cost))
    print(clf.get_best_model_name())
    mm = clf.get_best_model_inst()
    mm.fit(X)
    y_hat = mm.predict(X)
    print("yyyyyyyyyyy", eval_performance("silhouette_score", X=X, y_score=y_hat))

print("validates_records, ", validates_records)
print("test_records_s, ", test_records_s)
print("test_records_adj, ", test_records_adj)
print("time_cost_records, ", time_cost_records)
