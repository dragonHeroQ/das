import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import logging
import ray
from multiprocessing import Process
from automl.get_algorithm import (get_algorithm_by_key,
                                  get_all_classification_algorithm_keys,
                                  get_all_clustering_algorithm_keys,
                                  get_all_regression_algorithm_keys)
from sklearn.metrics import mean_squared_error
from airfoil.load_airfoil import load_airfoil
from libsvm_abalone.load_abalone import load_abalone
from superconduct.load_superconduct import load_superconduct
from cpusmall.load_cpusmall import load_cpusmall
from libsvm_cadata.load_cadata import load_cadata
from libsvm_space_ga.load_space_ga import load_space_ga
from mg.load_mg import load_mg

logger = logging.getLogger(__name__)

ray.init()

dataset = 'superconduct'

x_train, x_test, y_train, y_test = load_superconduct()

x_train_id = ray.put(x_train)
x_test_id = ray.put(x_test)
y_train_id = ray.put(y_train)
y_test_id = ray.put(y_test)


def getmbof(x):
	if isinstance(x, np.ndarray):
		return "{:.2f}MB".format(x.itemsize * x.size / 1048576.0)
	return "{:.2f}MB".format(sys.getsizeof(x) / 1048576.0)


@ray.remote
def run(x_train, x_test, y_train, y_test, algo1_key, algo2_key, concat_type='c'):

	cm1 = get_algorithm_by_key(algo1_key, random_state=0)
	cm1.fit(x_train, y_train)

	y_train_ba = cm1.predict(x_train)[:, np.newaxis]
	y_test_ba = cm1.predict(x_test)[:, np.newaxis]

	if concat_type == 'c':
		aug_x_train = np.hstack((x_train, y_train_ba))
		aug_x_test = np.hstack((x_test, y_test_ba))
	elif concat_type == 'p':
		aug_x_train = y_train_ba
		aug_x_test = y_test_ba
	elif concat_type == 'o':
		aug_x_train = x_train
		aug_x_test = x_test
	else:
		raise NotImplementedError

	if np.isnan(aug_x_train).any() or np.isnan(aug_x_test).any():
		logger.info("aug_x_* has NaN~~")
		return np.inf, np.inf

	print("aug_x_train and aug_x_test: ", getmbof(aug_x_train), getmbof(aug_x_test))
	# print("Creating {}...".format(algo2))
	cm2 = get_algorithm_by_key(algo2_key, random_state=0)
	cm2.fit(aug_x_train, y_train)
	# print("{} fitted".format(algo2))
	y_hat = cm2.predict(aug_x_test)
	y_hat_train = cm2.predict(aug_x_train)

	if np.isnan(y_hat).any():
		logger.info("y_hat has NaN~~")
		return np.inf, np.inf

	mse_train = mean_squared_error(y_train, y_hat_train)
	mse = mean_squared_error(y_test, y_hat)
	# print("CompModel on airfoil: {}".format(mse))
	return mse, mse_train


all_algos1 = get_all_regression_algorithm_keys()
all_algos1.remove("FMRegressor")
all_algos1.append("IdentityRegressor")
all_algos2 = get_all_regression_algorithm_keys()
all_algos2.remove("FMRegressor")

import time
import psutil
start_time = time.time()

mses = []
mse_dict = {}
for algo1_key in all_algos1:
	for algo2_key in all_algos2:
		time.sleep(10)
		print("Now Free Memory: ", psutil.virtual_memory().free)
		if algo1_key == 'IdentityRegressor':
			concat_type_candidates = ['o']
		else:
			concat_type_candidates = ['c', 'p']
		two_ids = []
		mapping_id_to_cattype = {}
		for concat_type in concat_type_candidates:
			MSE_id = run.remote(x_train_id,  x_test_id, y_train_id, y_test_id, algo1_key, algo2_key, concat_type)
			two_ids.append(MSE_id)
			mapping_id_to_cattype[MSE_id] = concat_type_candidates[len(two_ids)-1]
		ready_ids, unready_ids = ray.wait(two_ids, num_returns=2, timeout=40*1000)
		# print(ready_ids, unready_ids)
		if len(ready_ids) == 0:
			logger.info("Timeout Error")
			MSES = [np.inf, np.inf]
			print("{}+{}+{}: {}".format(algo1_key, algo2_key, 'c', 'inf'))
			mse_dict['{}+{}+{}'.format(algo1_key, algo2_key, 'c')] = 'inf'
			print("{}+{}+{}: {}".format(algo1_key, algo2_key, 'p', 'inf'))
			mse_dict['{}+{}+{}'.format(algo1_key, algo2_key, 'p')] = 'inf'
		else:
			MSES = ray.get(ready_ids)
			for i, ready_id in enumerate(ready_ids):
				cat_type = mapping_id_to_cattype[ready_id]
				print("{}+{}+{}: {}".format(algo1_key, algo2_key, cat_type, MSES[i]))
				mse_dict['{}+{}+{}'.format(algo1_key, algo2_key, cat_type)] = MSES[i]
			mses.extend(MSES)

import pickle

pickle.dump(mse_dict, open('{}_mses.pkl'.format(dataset), 'wb'))

sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1], reverse=False)
for k, v in sorted_mses[:10]:
	print(k, v)

end_time = time.time()
print("TimeCost = {}".format(end_time-start_time))

# import matplotlib.pyplot as plt
#
# plt.plot(mses)
# plt.show()

