import sys
sys.path.append('../../')
from automl.Regression.Regressor import Regressor
from sklearn.metrics.regression import mean_squared_error
import time
import numpy as np
from automl.performance_evaluation import eval_performance
from load_airfoil import load_airfoil
x_train, x_test, y_train, y_test = load_airfoil()
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

from sklearn.externals import joblib
for i in range(1):
    start_time = time.time()

    rgr = Regressor(q_epsilon=0.7, total_timebudget=6*60, per_run_timebudget=120,
                    validation_strategy="cv", validation_strategy_args=3,
                    use_HypTuner=True)

    rgr.version = i

    rgr.set_automl_mode(2)
    rgr.set_regression_mode(1)
    rgr.configure_default_algorithm_set()

    rgr.fit(x_train, y_train)

    try:
        y_hat = rgr.predict(x_test)

        time_cost = time.time() - start_time
        time_cost_records.append(time_cost)
        validates_records.append(rgr.get_best_val_score())

        tmp_res = eval_performance("mean_squared_error", y_true=y_test, y_score=y_hat)
        print(mean_squared_error(y_true=y_test, y_pred=y_hat))

        test_records.append(tmp_res)
        xxx.append(rgr.get_best_model_at_specify_timepointset(xx))
        print("automl_mode_2 on airfoil: ", tmp_res)
        print("Time_Cost: {}".format(time_cost))
    except:
        test_records[i] = -1
        xxx[i] = -1
        validates_records[i] = -1
        time_cost_records[i] = -1
        xx[i] = -1

rnd = 0
for i in xxx:
    #print("aaa")
    t_res = []
    print(i)
    for ii in i:
        try:
            if ii is None:
                t_res.append(0)
            else:
                if os.path.exists("model_%d_%d.pkl"%(rnd, ii)):
                    mod = joblib.load("model_%d_%d.pkl"%(rnd, ii))
                    y_hat = mod.predict(x_test)
                    tmp_res = eval_performance("mean_squared_error", y_true=y_test, y_score=y_hat)
                    t_res.append(tmp_res)
        except:
            pass
    rnd += 1
    print("每隔一分钟的性能in round %d"%rnd, t_res)
print("validates_records, ", validates_records)
print("test_records, ", test_records)
print("time_cost_records, ", time_cost_records)
