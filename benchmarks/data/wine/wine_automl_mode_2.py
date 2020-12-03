import sys
sys.path.append('../../')
import time
import warnings
#from load_dexter import load_dexter
from sklearn.model_selection import train_test_split
import sklearn.datasets
warnings.filterwarnings("ignore")

X, y = sklearn.datasets.load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(x_train), len(x_test))

from automl.Classification.classifier import Classifier
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
import time
from automl.performance_evaluation import eval_performance

import os
import logging
import automl
logger = logging.getLogger(automl.logger_name)
import warnings
warnings.filterwarnings("ignore")

NAME="wine"

validates_records = []
test_records = []
time_cost_records = []
xx = [(i+1) * 60 for i in range(60)]
xxx = []

from sklearn.externals import joblib
for i in range(1):

    start_time = time.time()

    clf = Classifier(total_timebudget=1.5*60, per_run_timebudget=60, random_state=i,
                     validation_strategy="cv", validation_strategy_args=3)
    clf.set_version(i)
    clf.configure_default_algorithm_set()

    clf.set_automl_mode(2)
    clf.set_classification_mode(1)
    # clf._fit_based_on_q_learning(, y)
    clf.fit(x_train, y_train)
    clf.refit(x_train, y_train)

    y_hat = clf.predict(x_test)



    time_cost = time.time() - start_time
    time_cost_records.append(time_cost)
    validates_records.append(clf.get_best_val_score())

    tmp_res = eval_performance("accuracy_score", y_true=y_test, y_score=y_hat)

    test_records.append(tmp_res)
    logger.info("automl_mode_2 on {}: {}".format(NAME, tmp_res))
    logger.info("Time_Cost: {}".format(time_cost))
    logger.info("Best Model: {}".format(clf.get_best_model_name()))


# rnd = 0
# for i in xxx:
#     #print("aaa")
#     t_res = []
#     print(i)
#     for ii in i:
#         try:
#             if ii is None:
#                 t_res.append(0)
#             else:
#                 if os.path.exists("model_%d_%d.pkl"%(rnd, ii)):
#                     mod = joblib.load("model_%d_%d.pkl"%(rnd, ii))
#                     y_hat = mod.predict(x_test)
#                     #print("hhh")
#                     # tmp_res = label_ranking_loss(y_true=y_test, y_score=y_hat)
#                     #print("aaa")
#                     tmp_res = eval_performance("accuracy_score", y_true=y_test, y_score=y_hat)
#                     t_res.append(tmp_res)
#         except:
#             pass
#     rnd += 1
#     logger.info("每隔一分钟的性能in round {}: {}".format(rnd, t_res))
logger.info("validates_records: {} ".format(validates_records))
logger.info("test_records: {}".format(test_records))
logger.info("time_cost_records: {}".format(time_cost_records))