from automl.HyperparameterOptimizer.USH import *
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../..")
import sklearn
import sklearn.datasets
from automl.compmodel import CompositeModel
from automl.HyperparameterOptimizer.USH import Bandit, USH
from automl.performance_evaluation import *
from letter.load_letter import load_letter


if __name__ == "__main__":

	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.SVM import SVC
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.RandomForest import RandomForest
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.ExtraTreesClassifier import ExtraTreesClassifier
	from automl.BaseAlgorithm.classification.XGBAlgorithm.xgb_classifier import XGBClassifier
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.Adaboost import AdaBoostClassifier
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.DA import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.LogisticRegression import LogisticRegression
	from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.NB import BernouliNB, GaussianNB, MultinomialNB

	# X_1, y_1 = sklearn.datasets.load_iris(return_X_y=True)
	# X, X_test, y, y_test = sklearn.model_selection.train_test_split(X_1, y_1, test_size=0.33, random_state=42)
	X, X_test, y, y_test = load_letter()

	evaluation_rule = 'accuracy_score'

	# bandit_inst = {
	# 	1: Bandit(algorithm_model=SVC()),
	# 	2: Bandit(algorithm_model=RandomForest()),
	# 	3: Bandit(algorithm_model=AdaBoostClassifier()),
	# 	4: Bandit(algorithm_model=QuadraticDiscriminantAnalysis()),
	# 	5: Bandit(algorithm_model=LinearDiscriminantAnalysis()),
	# 	6: Bandit(algorithm_model=LogisticRegression()),
	# 	7: Bandit(algorithm_model=BernouliNB()),
	# 	8: Bandit(algorithm_model=GaussianNB()),
	# }

	bandit_inst = {
		1: Bandit(
			algorithm_model=CompositeModel([("RandomForest", RandomForest()),
			                                ("RandomForest", RandomForest()), 'c'])),
		2: Bandit(
			algorithm_model=CompositeModel([("RandomForest", RandomForest()),
			                                ("XGBClassifier", XGBClassifier()), 'c'])),
		3: Bandit(
			algorithm_model=CompositeModel([("ExtraTreesClassifier", ExtraTreesClassifier()),
			                                ("RandomForest", RandomForest()), 'c'])),
		4: Bandit(
			algorithm_model=CompositeModel([("XGBClassifier", XGBClassifier()),
			                                ("RandomForest", RandomForest()), 'c'])),
		5: Bandit(
			algorithm_model=CompositeModel([("XGBClassifier", XGBClassifier()),
			                                ("XGBClassifier", XGBClassifier()), 'c'])),
		6: Bandit(
			algorithm_model=CompositeModel([("ExtraTreesClassifier", ExtraTreesClassifier()),
			                                ("ExtraTreesClassifier", ExtraTreesClassifier()), 'c'])),
		7: Bandit(
			algorithm_model=CompositeModel([("RandomForest", RandomForest()),
			                                ("ExtraTreesClassifier", ExtraTreesClassifier()), 'c'])),
		8: Bandit(
			algorithm_model=CompositeModel([("RandomForest", RandomForest()),
			                                ("LogisticRegression", LogisticRegression()), 'c'])),
	}

	ush_inst = USH(600, bandit_inst, budget_type="time", max_number_of_round=3,
	               evaluation_rule=evaluation_rule, worst_loss=None)

	ush_inst.run(X, y, random_state=0)

	print("ush best bandit", ush_inst.get_key_of_best_bandit())

	best_val_loss = None
	best_ind = -1
	best_para = None
	for i in bandit_inst.keys():
		print(i, bandit_inst[i].records)
		print(bandit_inst[i].get_mean(), bandit_inst[i].get_variation())
		print(i, bandit_inst[i].get_best_record(), bandit_inst[i].get_best_model_parameters())

		if best_val_loss is None or (bandit_inst[i].get_best_record() is not None
		                             and best_val_loss > bandit_inst[i].get_best_record()):
			best_val_loss = bandit_inst[i].get_best_record()
			best_ind = i
			best_para = bandit_inst[i].get_best_model_parameters()

	print("best val loss: ", best_val_loss)

	# refit
	bandit_inst[best_ind].algorithm_model.set_params(**best_para)
	bandit_inst[best_ind].algorithm_model.fit(X, y)

	print("test_score", eval_performance("accuracy_score", y_test, bandit_inst[best_ind].algorithm_model.predict(X_test)))

