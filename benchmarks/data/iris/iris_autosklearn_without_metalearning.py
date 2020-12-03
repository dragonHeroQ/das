import sys
sys.path.append('../../')
import sklearn.datasets
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
X, y = sklearn.datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(x_train), len(x_test))

RESULT = []

for i in range(10):
	start_time = time.time()

	from autosklearn.classification import AutoSklearnClassifier
	cls = AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=60,
                                initial_configurations_via_metalearning=0,
                                resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
                                ensemble_size=1, smac_scenario_args={"initial_incumbent":"RANDOM"})
	cls.fit(x_train, y_train)
	cls.refit(x_train, y_train)
	y_hat = cls.predict(x_test)

	time_cost = time.time() - start_time
	acc = accuracy_score(y_test, y_hat)
	print("(600/15/cv3) AutoSklearn on digits: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))
	# AutoSklearn on digits:  0.94
	# Time Cost: 356.16118597984314

for res in RESULT:
    print("{}, cost {} s".format(res[0], res[1]))
