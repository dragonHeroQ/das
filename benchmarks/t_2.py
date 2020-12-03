import sys
import copy
from das.util.common_utils import kill_tree
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


algo_space = [GradientBoostingClassifier(), RandomForestClassifier(),
              SVC(), AdaBoostClassifier(), ExtraTreesClassifier(),
              BernoulliNB(), MultinomialNB(), GaussianNB(),
              MLPClassifier(), KNeighborsClassifier(), XGBClassifier(), LGBMClassifier(),
              ]


def process(X, y, estimators, return_dict):
	ray.init(redis_address="192.168.100.35:6379")
	if not isinstance(X, ray.ObjectID):
		X = ray.put(X)
		return_dict['Xid'] = X

	coha = [fit.remote(est, X, y) for est in estimators]
	coha_res = ray.get(coha)
	print(coha_res)


if __name__ == '__main__':
	X, y = load_digits(return_X_y=True)
	import multiprocessing

	mgr = multiprocessing.Manager()
	return_dict = mgr.dict()

	for ik in range(100):
		print("Running {}".format(ik))
		len_algo_space = len(algo_space)
		res = np.random.choice(range(len_algo_space), 4)
		estimators = list(map(lambda x: algo_space[x], res))
		for kk in estimators:
			print(kk.__class__)
		if 'Xid' in return_dict:
			print("Xid REUSE...")
			X = return_dict['Xid']
		p = multiprocessing.Process(target=process,
		                            args=(X, y, estimators, return_dict))
		print("daemon: ", p.daemon)
		p.start()
		p.join(30)

		if p.is_alive():
			print("Additional Terminating...")
			p.terminate()
			kill_tree(p.pid)

