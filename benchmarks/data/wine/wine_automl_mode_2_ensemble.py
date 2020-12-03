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

    clf = Classifier(total_timebudget=2*60, per_run_timebudget=60, ensemble_size=25,
                     validation_strategy="cv", validation_strategy_args=3, random_state=42+i)
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


logger.info("validates_records: {} ".format(validates_records))
logger.info("test_records: {}".format(test_records))
logger.info("time_cost_records: {}".format(time_cost_records))