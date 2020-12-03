import sys
sys.path.append('../../')
import sklearn.datasets
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
print(len(x_train), len(x_test))

RESULT = []

for i in range(5):
	start_time = time.time()

	from autosklearn.classification import AutoSklearnClassifier

	cls = AutoSklearnClassifier(time_left_for_this_task=1200, per_run_time_limit=15,
	                            resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
	                            ensemble_size=1, include_estimators=["libsvm_svc"],
	                            initial_configurations_via_metalearning=0)
	# component list:
	# ["adaboost", "bernoulli_nb", "decision_tree", "extra_trees", "gaussian_nb", "gradient_boosting",
	# "k_nearest_neighbors", "lda", "liblinear_svc", "libsvm_svc",
	# "multinomial_nb", "passive_aggressive", "qda", "random_forest", "sgd", "xgradient_boosting"]

	cls.fit(x_train, y_train)
	cls.refit(x_train, y_train)
	y_hat = cls.predict(x_test)

	time_cost = time.time() - start_time
	acc = accuracy_score(y_test, y_hat)
	print("AutoSklearn on Breast Cancer: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))
	# AutoSklearn libsvm_SVC on Breast Cancer:  0.973404255319149
	# Time Cost: 355.02907633781433

	# AutoSklearn libsvm_SVC on Breast Cancer:  0.9468085106382979
	# Time Cost: 18.870484590530396

for res in RESULT:
	print("{}, cost {} s".format(res[0], res[1]))
