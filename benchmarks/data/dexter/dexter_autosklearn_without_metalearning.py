import sys
sys.path.append('../../')
import time
from sklearn.metrics.classification import accuracy_score
from load_dexter import load_dexter

x_train, x_test, y_train, y_test = load_dexter()
print(len(x_train), len(x_test))

RESULT = []

for i in range(10):

	start_time = time.time()

	from autosklearn.classification import AutoSklearnClassifier
	cls = AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=60, initial_configurations_via_metalearning=0,
								resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
	                            ensemble_size=1, smac_scenario_args={"initial_incumbent":"RANDOM"})

	cls.fit(x_train, y_train)
	cls.refit(x_train, y_train)
	y_hat = cls.predict(x_test)

	time_cost = time.time() - start_time
	acc = accuracy_score(y_test, y_hat)
	print("AutoSklearn on gisette: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))


for res in RESULT:
	print("{}, cost {} s".format(res[0], res[1]))
