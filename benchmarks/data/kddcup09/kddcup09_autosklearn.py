import sys
sys.path.append('../../')
import time
from sklearn.metrics.classification import accuracy_score
from load_kddcup09 import load_kddcup09

x_train, x_test, y_train, y_test = load_kddcup09()
print(len(x_train), len(x_test))
print(len(x_train[0]))
RESULT = []

for i in range(10):

	start_time = time.time()

	from autosklearn.classification import AutoSklearnClassifier
	cls = AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=60,
								resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
	                            ensemble_size=1)

	cls.fit(x_train, y_train)
	cls.refit(x_train, y_train)
	y_hat = cls.predict(x_test)

	time_cost = time.time() - start_time
	acc = accuracy_score(y_test, y_hat)
	print("AutoSklearn on kddcup09: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))


for res in RESULT:
	print("{}, cost {} s".format(res[0], res[1]))
