import sys
sys.path.append('../../')
import time
import warnings
#from load_dexter import load_dexter
from sklearn.model_selection import train_test_split
import sklearn.datasets
warnings.filterwarnings("ignore")

X, y = sklearn.datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(len(x_train), len(x_test))

from automl.Classification.classifier import Classifier
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
test_records = []
time_cost_records = []
xx = [(i+1) * 1 for i in range(60)]
xxx = []

from sklearn.externals import joblib
for i in range(1):

    start_time = time.time()

    clf = Classifier(total_timebudget=60*1, per_run_timebudget=10,
                     validation_strategy="cv", validation_strategy_args=3)
    clf.set_version(i)
    clf.configure_default_algorithm_set()

    clf.set_automl_mode(1)
    clf.set_classification_mode(1)
    # clf._fit_based_on_q_learning(, y)
    clf.fit(x_train, y_train)

    y_hat = clf.predict(x_test)



    time_cost = time.time() - start_time
    time_cost_records.append(time_cost)
    validates_records.append(clf.get_best_val_score())

    tmp_res = eval_performance("accuracy_score", y_true=y_test, y_score=y_hat)

    test_records.append(tmp_res)
    xxx.append(clf.get_best_model_at_specify_timepointset(xx))
    print("automl_mode_2 on breast_cancer: ", tmp_res)
    print("Time_Cost: {}".format(time_cost))
    print(clf.get_best_model_name())

    clf.print_time_used_by_algorithm()
    clf.print_record_for_every_algorithm()


# rnd = 0
# for i in xxx:
#     #print("aaa")
#     t_res = []
#     print(i)
#     for ii in i:
#         if ii is None:
#             t_res.append(0)
#         else:
#             if os.path.exists("model_%d_%d.pkl"%(rnd, ii)):
#                 mod = joblib.load("model_%d_%d.pkl"%(rnd, ii))
#                 y_hat = mod.predict(x_test)
#                 tmp_res = eval_performance("accuracy_score", y_true=y_test, y_score=y_hat)
#                 t_res.append(tmp_res)
#     rnd += 1
#     print("每隔一分钟的性能in round %d"%rnd, t_res)

print("validates_records, ", validates_records)
print("test_records, ", test_records)
print("time_cost_records, ", time_cost_records)