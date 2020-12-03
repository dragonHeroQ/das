import sys
import copy
from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
from das.BaseAlgorithm.Classification.XGBClassifier import XGBClassifier
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_digits
from data.yeast.load_yeast import load_yeast
import numpy as np
import ray


@ray.remote
def fit(est, X, y):
	est.fit(X, y)
	return est.predict(X)


@ray.remote
def fit_2(est, X, y, Xcat):
	X = np.hstack((X, Xcat))
	a, b, c = copy.deepcopy(est), copy.deepcopy(est), copy.deepcopy(est)
	b.n_estimators = b.n_estimators + 80
	c.n_estimators = c.n_estimators + 90
	ests = [a, b, c]
	res = [fit.remote(e, X, y) for e in ests]
	return ray.get(res)


def process(X, y, Xcat, n_est, return_dict):
	ray.init(redis_address="192.168.100.35:6379", ignore_reinit_error=True)
	if not isinstance(X, ray.ObjectID):
		X = ray.put(X)
		return_dict['Xid'] = X
	estimators = [RandomForestClassifier(n_estimators=n_est),
	              ExtraTreesClassifier(n_estimators=n_est),
	              XGBClassifier(n_estimators=n_est),
	              ]

	coha = [fit_2.remote(est, X, y, Xcat) for est in estimators]
	coha_res = ray.get(coha)
	print(coha_res)


if __name__ == '__main__':
	X, _, y, _ = load_yeast()
	import multiprocessing

	ray.init(redis_address="192.168.100.35:6379")
	X = ray.put(X)
	n = y.shape[0]
	Xcat = np.zeros((n, 20))
	Xcat = ray.put(Xcat)
	y = ray.put(y)
	mgr = multiprocessing.Manager()
	return_dict = mgr.dict()
	p = multiprocessing.Process(target=process,
	                            args=(X, y, Xcat, 100, return_dict))
	p.start()
	p.join(30)

	if 'Xid' in return_dict:
		print("Xid REUSE...")
		X = return_dict['Xid']

	p = multiprocessing.Process(target=process,
	                            args=(X, y, Xcat, 200, return_dict))
	p.start()
	p.join(30)

