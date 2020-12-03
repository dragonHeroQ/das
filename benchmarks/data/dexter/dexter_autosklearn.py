import sys
sys.path.append('../../')
import time
from sklearn.metrics.classification import accuracy_score
from load_dexter import load_dexter
from automl.crossvalidate import cross_validate_score
x_train, x_test, y_train, y_test = load_dexter()
print(len(x_train), len(x_test))

RESULT = []

for i in range(1):

	start_time = time.time()

	from autosklearn.classification import AutoSklearnClassifier
	from autosklearn.pipeline.components.classification.random_forest import RandomForest
	cls = AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60, initial_configurations_via_metalearning=0,
								resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
	                            ensemble_size=1)



	cls.fit(x_train, y_train)
	print(cls.show_models())

	#print(cross_validate_score(cls, X=x_train, y=y_train, cv=3))

	cls.refit(x_train, y_train)
	print(cls.show_models())




	y_hat = cls.predict(x_test)

	time_cost = time.time() - start_time
	acc = accuracy_score(y_test, y_hat)
	print("AutoSklearn on gisette: ", acc)
	print("Time Cost: {}".format(time_cost))
	RESULT.append((acc, time_cost))


for res in RESULT:
	print("{}, cost {} s".format(res[0], res[1]))
