import sys
sys.path.append('../../')

from automl.Regression.Regressor import Regressor

from automl.get_algorithm import get_algorithm_by_key

from automl.Regression.Regressor import Regressor
from sklearn.metrics.regression import mean_squared_error
import time
import numpy as np
from automl.performance_evaluation import eval_performance
from load_superconduct import load_superconduct
x_train, x_test, y_train, y_test = load_superconduct()
print("x_train shape: {}".format(np.shape(x_train)))
print("x_test shape: {}".format(np.shape(x_test)))
print("y_train shape: {}".format(np.shape(y_train)))
print("y_test shape: {}".format(np.shape(y_test)))

import os
import warnings
warnings.filterwarnings("ignore")

validates_records = []
test_records = []
time_cost_records = []
xx = [(i+1) * 60 for i in range(60)]
xxx = []

from automl.Regression.Regressor import logger

from sklearn.externals import joblib
for i in range(1):
    start_time = time.time()

    # rgr = Regressor(q_epsilon=0.7, total_timebudget=60*10, per_run_timebudget=60,
    #                  validation_strategy="cv", validation_strategy_args=3)

    # rgr = Regressor_new.Regressor_new(q_epsilon=0.7, total_timebudget=60 * 60, per_run_timebudget=60,
    #                                   validation_strategy="cv", validation_strategy_args=3, random_state=i,
    #                                   log_level='INFO')

    rgr = Regressor(q_epsilon=0.5, total_timebudget=3 * 60, per_run_timebudget=60, ensemble_size=25,
                        validation_strategy="cv", validation_strategy_args=3, random_state=i, log_level="INFO")

    rgr.version = i

    rgr.set_automl_mode(2)
    rgr.set_regression_mode(1)
    rgr.configure_default_algorithm_set()
    # rgr.addRegressor({'ARDRegression': ARDRegression.ARDRegression()})

    # rgr.addRegressor({'AdaboostRegressor': get_algorithm_by_key('AdaboostRegressor')})
    # rgr.addFeatureSelector({'NMF': get_algorithm_by_key('NMF')})

    rgr.fit(x_train, y_train)
    rgr.refit(x_train, y_train)

    try:
        y_hat = rgr.predict(x_test)
        time_cost = time.time() - start_time
        time_cost_records.append(time_cost)
        validates_records.append(rgr.get_best_val_score())

        tmp_res = eval_performance("mean_squared_error", y_true=y_test, y_score=y_hat)
        # print(mean_squared_error(y_true=y_test, y_pred=y_hat))

        test_records.append(tmp_res)
        xxx.append(rgr.get_best_model_at_specify_timepointset(xx))
        logger.info("automl_mode_2 on superconduct: {}".format(tmp_res))
        logger.info("Time_Cost: {}".format(time_cost))
    except:
        test_records[i] = -1
        xxx[i] = -1
        validates_records[i] = -1
        time_cost_records[i] = -1
        xx[i] = -1


logger.info("validates_records: {}".format(validates_records))
logger.info("test_records: {}".format(test_records))
logger.info("time_cost_records: {}".format(time_cost_records))