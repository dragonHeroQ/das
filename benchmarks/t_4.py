import sys
import copy
from das.util.common_utils import kill_tree
from das.BaseAlgorithm.ray_utils import unit_fit_predict_kfold
from das.BaseAlgorithm.algorithm_space import get_algorithm_class_by_key
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import numpy as np
import ray


@ray.remote
def fit(est, X, y):
	est.fit(X, y)
	return est.predict(X)


algo_space = ["SGDClassifier",
                                       "GBDTClassifier",
                                       "LogisticRegression",
                                       "DecisionTreeClassifier",
                                       "RandomForestClassifier",
                                       "SVC",
                                       "AdaboostClassifier",
                                       "LinearDiscriminantAnalysis",
                                       "QuadraticDiscriminantAnalysis",
                                       "ExtraTreesClassifier",
                                       "BernoulliNB",
                                       "MultinomialNB",
                                       "GaussianNB",
                                       # "GPClassifier",
                                       "MLPClassifier",
                                       "KNeighborsClassifier",
                                       # "RadiusNeighborsClassifier",
                                       "XGBClassifier",
                                       "LGBClassifier",
                                       # "IdentityClassifier"
                                       ]


def process(X, y, estimators, return_dict):
	if not ray.is_initialized():
		ray.init(redis_address="192.168.100.35:6379")
	if not isinstance(X, ray.ObjectID):
		X = ray.put(X)
		return_dict['Xid'] = X
	Xcat = np.zeros((y.shape[0], 10))
	coha = [unit_fit_predict_kfold.remote(est, X, y, Xcat, 3, 'predict_proba',
	                                      'classification', None, 3) for est in estimators]
	coha_res = ray.get(coha)
	print(coha_res)


if __name__ == '__main__':
	X, y = load_digits(return_X_y=True)
	from multiprocessing import Pool
	import multiprocessing
	ps = Pool(1)

	mgr = multiprocessing.Manager()
	return_dict = mgr.dict()

	for ik in range(100):
		print("Running {}".format(ik))
		len_algo_space = len(algo_space)
		res = np.random.choice(range(len_algo_space), 4)
		estimators = list(map(lambda x: get_algorithm_class_by_key(algo_space[x])(), res))
		for kk in estimators:
			print(kk.__class__)
		if 'Xid' in return_dict:
			print("Xid REUSE...")
			X = return_dict['Xid']

		ans = ps.apply_async(process, args=(X, y, estimators, return_dict))  # 异步执行
		print(ans.get(timeout=30))

	ps.terminate()
	ps.join()
