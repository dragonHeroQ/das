import sys
sys.path.append('../../')
import sklearn.datasets
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
X, y = sklearn.datasets.load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
print(len(x_train), len(x_test))

RESULT = []

for i in range(5):
    start_time = time.time()

    from autosklearn.classification import AutoSklearnClassifier
    cls = AutoSklearnClassifier(time_left_for_this_task=1200, per_run_time_limit=15,
                                resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
                                ensemble_size=1, exclude_estimators=['xgradient_boosting'])
    cls.fit(x_train, y_train)
    cls.refit(x_train, y_train)
    y_hat = cls.predict(x_test)

    time_cost = time.time() - start_time
    acc = accuracy_score(y_test, y_hat)
    print("AutoSklearn on wine: ", acc)
    print("Time Cost: {}".format(time_cost))
    RESULT.append((acc, time_cost))
    # AutoSklearn on wine:  1.0
    # Time Cost: 355.61260056495667

    # AutoSklearn on wine:  0.9491525423728814
    # Time Cost: 115.00382018089294

for res in RESULT:
    print("{}, cost {} s".format(res[0], res[1]))
