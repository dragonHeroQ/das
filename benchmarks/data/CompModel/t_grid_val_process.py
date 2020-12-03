import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import logging
import importlib
from multiprocessing import Process, Pool
import multiprocessing
from automl.compmodel import CompositeModel
from automl.get_algorithm import (get_algorithm_by_key,
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


def kill_tree(pid, including_parent=True):
	parent = psutil.Process(pid)
	for child in parent.children(recursive=True):
		print("child", child)
		child.kill()

	if including_parent:
		parent.kill()


def getmbof(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def get_timeout(size_x_train, size_x_test):
	total_size = size_x_train + size_x_test
	return total_size


def sampling_fidelity(X, mb=1.0):
	low = 1
	high = X.shape[0]
	while low <= high:
		mid = (low + high) // 2
		if getmbof(X[:mid]) < mb:
			low = mid + 1
		else:
			high = mid - 1
	return high


if len(sys.argv) > 1:
	data_name = sys.argv[1]
else:
	data_name = 'cadata'
print("Data Name: {}".format(data_name))
if data_name in ['abalone', 'cadata', 'space_ga']:
	data_dir = 'libsvm_{}'.format(data_name)
else:
	data_dir = data_name
load_xx = importlib.import_module('{}.load_{}'.format(data_dir, data_name))
x_train, x_test, y_train, y_test = getattr(load_xx, 'load_{}'.format(data_name))()

if len(sys.argv) > 2:
	rng = int(sys.argv[2])
else:
	rng = 0
print("RNG = {}".format(rng))

if data_name in ['superconduct', 'cadata']:
	fidelity = sampling_fidelity(X=x_train, mb=1.0)
	print("Proper Fidelity: {}".format(fidelity))
	np.random.seed(rng)
	indexes = np.random.choice(len(x_train), fidelity)
	x_train_fidelity = x_train[indexes]
	y_train_fidelity = y_train[indexes]
	# x_train_fidelity = x_train[:fidelity]
	# y_train_fidelity = y_train[:fidelity]
else:
	x_train_fidelity = x_train
	y_train_fidelity = y_train

print("x_train size, x_train_fidelity size = {:.2f} MB, {:.2f} MB".format(getmbof(x_train), getmbof(x_train_fidelity)))


def run(X, y, in_comp_model):
	# idx_model = in_comp_model[0]
	evaluation_rule = 'mean_squared_error'
	comp_model = CompositeModel([(in_comp_model[1], get_algorithm_by_key(in_comp_model[1], random_state=rng)),
	                             (in_comp_model[2], get_algorithm_by_key(in_comp_model[2], random_state=42+rng)),
	                             in_comp_model[3]])

	reward = comp_model.compute(X=X, y=y,
			                    evaluation_rule=evaluation_rule,
			                    validation_strategy_args=3,
	                            random_state=23+rng*23)
	return reward['info']['val_{}'.format(evaluation_rule)]


def get_a_model(model_space: dict):
	sorted_model_space = sorted(model_space.items(),
	                            key=lambda x: algorithm_scores_1[x[0][1]]*algorithm_scores_2[x[0][2]],
	                            reverse=True)
	chosed_model = sorted_model_space[0][0]
	return chosed_model, algorithm_scores_1[chosed_model[1]]*algorithm_scores_2[chosed_model[2]]


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
idx = 1
for algo1_key in all_algos1:
	for algo2_key in all_algos2:
		if algo1_key == 'IdentityRegressor':
			concat_type_candidates = ['o']
		else:
			concat_type_candidates = ['c', 'p']
		for concat_type in concat_type_candidates:
			composite_models_space[(idx, algo1_key, algo2_key, concat_type)] = 1.0
			idx += 1

# composite_models_space = {(1, 'IdentityRegressor', 'ExtraTreesRegressor', 'o'): 1.0}
# composite_models_space = {(1, 'RadiusNeighborsRegressor', 'XGBRegressor', 'c'): 1.0}
# composite_models_space = {(1, 'ARDRegression', 'ExtraTreesRegressor', 'c'): 1.0}

algorithm_scores_1 = {}
for algo1 in all_algos1:
	algorithm_scores_1[algo1] = 1.0
algorithm_scores_2 = {}
for algo2 in all_algos2:
	algorithm_scores_2[algo2] = 1.0

mses = []
mse_dict = {}
pool = Pool(1)
Time_Costs = [10, ]
gamma1 = 0.95
gamma2 = 0.99
ran_composite_models = {}
num_composite_models = len(composite_models_space)
first_model_score_less_1 = True

# print(composite_models_space.keys())
print("Total Composite Models: {}".format(num_composite_models))

while len(composite_models_space) > 0:
	pool = Pool(1)
	comp_model, model_score = get_a_model(composite_models_space)
	composite_models_space.pop(comp_model)
	free_memory = (psutil.virtual_memory().free + psutil.virtual_memory().cached) / 1048576.0
	if len(composite_models_space) % 10 == 0:
		print("Now Free Memory: {} MB".format(free_memory))
	if free_memory < minimum_free_memory:
		print("MemoryOut! Breaking...")
		break
	if first_model_score_less_1 and model_score < 1.0:
		first_model_score_less_1 = False
		print("{}/{} Model Score becomes <1.0, Starting [mu+sigma] timeout !".format(len(ran_composite_models),
		                                                                             num_composite_models))
	# Time Out should be 10 ~ 360
	if model_score < 1.0:
		running_timeout = max(10.0, float(np.mean(Time_Costs) + np.std(Time_Costs)))
	else:
		running_timeout = (np.mean(Time_Costs) + np.std(Time_Costs)) * 10
	timeout = min(360.0, float(running_timeout))
	print("== Time Out: {:.1f} s".format(timeout))
	print(">> Running {}+{}+{}+{}".format(comp_model[0], comp_model[1], comp_model[2], comp_model[3]))
	inner_start_time = time.time()
	try:
		result_future = pool.apply_async(run,
		                                 args=(x_train_fidelity, y_train_fidelity, comp_model))

		MSE_val = result_future.get(timeout=timeout)
		inner_time_cost = time.time() - inner_start_time
		Time_Costs.append(inner_time_cost)
	except multiprocessing.context.TimeoutError as e:
		# print("multiprocessing.context.TimeoutError")
		print("TimeOut Error")
		MSE_val = np.inf
		algorithm_scores_1[comp_model[1]] *= gamma1
		algorithm_scores_2[comp_model[2]] *= gamma1
	except Exception as e:
		print("Other Exceptions")
		print(e)
		MSE_val = np.inf
		algorithm_scores_1[comp_model[1]] *= gamma2
		algorithm_scores_2[comp_model[2]] *= gamma2
	finally:
		if MSE_val == np.inf:
			inner_time_cost = time.time() - inner_start_time
			# Time_Costs.append(float(np.mean(Time_Costs)) + 1)

	print("{}+{}+{}+{}: val={}, timecost={}".format(comp_model[0], comp_model[1], comp_model[2], comp_model[3],
	                                                MSE_val, inner_time_cost))
	mse_dict['{}+{}+{}+{}'.format(comp_model[0], comp_model[1],
	                              comp_model[2], comp_model[3])] = (MSE_val, inner_time_cost)
	mses.append((MSE_val, inner_time_cost))
	ran_composite_models[comp_model] = MSE_val
	pool.terminate()
	pool.close()

end_time = time.time()
mse_dict['Time_Cost'] = end_time-start_time
print("TimeCost = {}".format(end_time-start_time))

# print(mse_dict)

import pickle

pickle.dump(mse_dict, open('{}_mses_{}.pkl'.format(data_name, rng), 'wb'))

# f_perf = open('performance0110.txt', 'a')
# f_perf.write("\nrng = {}\n".format(rng))
# sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1], reverse=False)
# for k, v in sorted_mses[:10]:
# 	if k == 'Time_Cost':
# 		continue
# 	idx, algo1, algo2, cat_type = k.split('+')
# 	comp = CompositeModel([(algo1, get_algorithm_by_key(algo1, random_state=rng)),
# 	                       (algo2, get_algorithm_by_key(algo2, random_state=42+rng)),
# 						   cat_type])
# 	evaluation_rule = 'mean_squared_error'
# 	manager = multiprocessing.Manager()
# 	d = manager.dict()
# 	p = Process(target=comp.score, args=(x_train, y_train, x_test, y_test, evaluation_rule, 23*rng),
# 	            kwargs={"return_dict": d})
# 	p.start()
# 	p.join(360)
# 	if p.is_alive():
# 		kill_tree(p.pid)
# 	score = np.inf
# 	if 'test_score' in d.keys():
# 		score = d["test_score"]
# 	print(k, v[0], score, v[1])
# 	f_perf.write("{}, {}, {}, {}\n".format(k, v[0], score, v[1]))
#
# f_perf.close()


