import sys
sys.path.append('../../')
import time
from sklearn.metrics.classification import accuracy_score, f1_score, hamming_loss
from sklearn.metrics.ranking import label_ranking_loss
from load_scene import load_scene

x_train, x_test, y_train, y_test = load_scene()
print(len(x_train), len(x_test))

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
	acc = label_ranking_loss(y_test, y_hat)
	print("AutoSklearn on yeast: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))

# label_ranking_loss: 0.655737172116281

for res in RESULT:
	print("{}, cost {} s".format(res[0], res[1]))
