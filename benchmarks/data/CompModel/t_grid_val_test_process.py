import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import logging
from multiprocessing import Process, Pool
import multiprocessing
from automl.compmodel import CompositeModel
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

dataset = 'superconduct'

x_train, x_test, y_train, y_test = load_superconduct()


def getmbof(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def get_timeout(size_x_train, size_x_test):
	total_size = size_x_train + size_x_test
	return total_size


print("x_train size, x_test size = {:.2f} MB, {:.2f} MB".format(getmbof(x_train), getmbof(x_test)))


def run(x_train, x_test, y_train, y_test, algo1_key, algo2_key, concat_type='c'):
	evaluation_rule = 'mean_squared_error'
	comp_model = CompositeModel([get_algorithm_by_key(algo1_key, random_state=0),
	                             get_algorithm_by_key(algo2_key, random_state=42),
	                             concat_type])

	reward = comp_model.compute(X=x_train, y=y_train,
			                    evaluation_rule=evaluation_rule,
			                    validation_strategy_args=3,
	                            random_state=23)

	test_score = comp_model.score(X=x_train, y=y_train, x_test=x_test, y_test=y_test, evaluation_rule=evaluation_rule)

	return reward['info']['val_{}'.format(evaluation_rule)], test_score


def get_a_model(model_space: dict):
	sorted_model_space = sorted(model_space.items(),
	                            key=lambda x: algorithm_scores_1[x[0][0]]*algorithm_scores_2[x[0][1]],
	                            reverse=True)
	return sorted_model_space[0][0]


all_algos1 = get_all_regression_algorithm_keys()
all_algos1.remove("FMRegressor")
all_algos1.append("IdentityRegressor")
all_algos2 = get_all_regression_algorithm_keys()
all_algos2.remove("FMRegressor")

import time
import psutil
total_memory = psutil.virtual_memory().total / 1048576.0
minimum_free_memory = 1.0 / 16.0 * total_memory
start_time = time.time()

composite_models_space = {}
for algo1_key in all_algos1:
	for algo2_key in all_algos2:
		if algo1_key == 'IdentityRegressor':
			concat_type_candidates = ['o']
		else:
			concat_type_candidates = ['c', 'p']
		for concat_type in concat_type_candidates:
			composite_models_space[(algo1_key, algo2_key, concat_type)] = 1.0

# composite_models_space = {('ExtraTreesRegressor', 'ExtraTreesRegressor', 'c'): 1.0}
# composite_models_space = {('AdaboostRegressor', 'ARDRegression', 'c'): 1.0}

algorithm_scores_1 = {}
for algo1 in all_algos1:
	algorithm_scores_1[algo1] = 1.0
algorithm_scores_2 = {}
for algo2 in all_algos2:
	algorithm_scores_2[algo2] = 1.0

mses = []
mse_dict = {}
pool = Pool(1)
Time_Costs = [5, ]
gamma1 = 0.95
gamma2 = 0.99
ran_composite_models = []

while len(composite_models_space) > 0:
	pool = Pool(1)
	comp_model = get_a_model(composite_models_space)
	composite_models_space.pop(comp_model)
	ran_composite_models.append(comp_model)
	free_memory = (psutil.virtual_memory().free + psutil.virtual_memory().cached) / 1048576.0
	print("Now Free Memory: {} MB".format(free_memory))
	if free_memory < minimum_free_memory:
		print("MemoryOut!")
		exit(0)
	timeout = min(360.0, 42 * float(np.mean(Time_Costs)))
	print("== Time Out: {:.1f} s".format(timeout))
	inner_start_time = time.time()
	try:
		result_future = pool.apply_async(run,
		                                 args=(x_train, x_test, y_train, y_test,
		                                       comp_model[0], comp_model[1], comp_model[2]))

		MSE_val, MSE_test = result_future.get(timeout=timeout)
		inner_time_cost = time.time() - inner_start_time
		Time_Costs.append(inner_time_cost)
	except multiprocessing.context.TimeoutError as e:
		# print("multiprocessing.context.TimeoutError")
		print("TimeOut Error")
		MSE_val, MSE_test = np.inf, np.inf
		algorithm_scores_1[comp_model[0]] *= gamma1
		algorithm_scores_2[comp_model[1]] *= gamma1
	except Exception as e:
		print("Other Exceptions")
		print(e)
		MSE_val, MSE_test = np.inf, np.inf
		algorithm_scores_1[comp_model[0]] *= gamma2
		algorithm_scores_2[comp_model[1]] *= gamma2
	finally:
		print("Finally, MSE_val = {}".format(MSE_val))
		if MSE_val == np.inf:
			inner_time_cost = time.time() - inner_start_time
			# Time_Costs.append(float(np.mean(Time_Costs)) + 1)

	print("{}+{}+{}: val={}, test={}, timecost={}".format(comp_model[0], comp_model[1], comp_model[2],
	                                                      MSE_val, MSE_test, inner_time_cost))
	mse_dict['{}+{}+{}'.format(comp_model[0], comp_model[1], comp_model[2])] = (MSE_val, MSE_test, inner_time_cost)
	mses.append((MSE_val, MSE_test, inner_time_cost))
	pool.terminate()
	pool.close()

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

