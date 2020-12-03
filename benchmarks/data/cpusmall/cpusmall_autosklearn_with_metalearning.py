import sys
sys.path.append('../../')
import time
from load_cpusmall import load_cpusmall
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test =load_cpusmall()

RESULT = []

for i in range(10):
    start_time = time.time()
    from autosklearn.regression import AutoSklearnRegressor

    rgr = AutoSklearnRegressor(time_left_for_this_task=3600, per_run_time_limit=120,
                               resampling_strategy="cv", resampling_strategy_arguments={'folds': 3}, ensemble_size=1,
                               )

    rgr.fit(x_train, y_train)
    rgr.refit(x_train, y_train)

    y_hat = rgr.predict(x_test)

    time_cost = time.time() - start_time
    mse = mean_squared_error(y_test, y_hat)
    print("Autosklearn on cpusmall: {}".format(mse))
    print("Time Cost: {}".format(time_cost))
    RESULT.append((mse, time_cost))

for res in RESULT:
    print("{}, cost {} s".format(res[0], res[1]))