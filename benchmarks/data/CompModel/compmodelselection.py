import sys
import pickle
import importlib
sys.path.append("../")
sys.path.append("../..")
from automl.HyperparameterOptimizer.USH import Bandit, USH
from automl.performance_evaluation import *
from letter.load_letter import load_letter
from automl.compmodel import CompositeModel
from automl.get_algorithm import (get_algorithm_by_key, get_all_regression_algorithm_keys)
from sklearn.metrics import mean_squared_error
from airfoil.load_airfoil import load_airfoil
from libsvm_abalone.load_abalone import load_abalone
from superconduct.load_superconduct import load_superconduct
from cpusmall.load_cpusmall import load_cpusmall
from libsvm_cadata.load_cadata import load_cadata
from libsvm_space_ga.load_space_ga import load_space_ga
from mg.load_mg import load_mg


def getmbof(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def get_timeout(size_x_train, size_x_test):
	total_size = size_x_train + size_x_test
	return total_size


def get_1MB_fidelity(X):
	low = 1
	high = X.shape[0]
	while low <= high:
		mid = (low + high) // 2
		if getmbof(X[:mid]) < 1.0:
			low = mid + 1
		else:
			high = mid - 1
	return high


if len(sys.argv) > 1:
	data_name = sys.argv[1]
else:
	data_name = 'superconduct'
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

if len(sys.argv) > 3:
	time_budget = int(sys.argv[3])
else:
	time_budget = 3600
print("Total Time Budget = {}".format(time_budget))

if data_name == 'superconduct':
	fidelity = get_1MB_fidelity(X=x_train)
	print("Proper Fidelity: {}".format(fidelity))
	x_train_fidelity = x_train[:fidelity]
	y_train_fidelity = y_train[:fidelity]
else:
	x_train_fidelity = x_train
	y_train_fidelity = y_train

print("x_train size, x_train_fidelity size = {:.2f} MB, {:.2f} MB".format(getmbof(x_train), getmbof(x_train_fidelity)))


if __name__ == "__main__":

	evaluation_rule = 'mean_squared_error'

	mse_dict = pickle.load(open('../CompModel_PKLS/{}_mses_{}.pkl'.format(data_name, rng), 'rb'))
	mse_dict.pop('Time_Cost')

	sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1][0], reverse=False)
	sorted_mses = sorted_mses[:max(1, int(0.2*len(sorted_mses)))]  # 20-80 rule

	bandit_inst = {}
	idx = 0
	worst_loss = 0
	for comp_model, (val_loss, time_cost) in sorted_mses:
		if val_loss != np.inf:
			m_idx, cm1, cm2, cat_type = comp_model.split('+')
			default_comp_model = CompositeModel([(cm1, get_algorithm_by_key(cm1, random_state=rng)),
			                                     (cm2, get_algorithm_by_key(cm2, random_state=42+rng)),
			                                     cat_type])
			bandit_inst[idx] = Bandit(algorithm_model=default_comp_model)
			bandit_inst[idx].add_records([val_loss, ], [default_comp_model.get_params()])

			idx += 1
			# if worst loss not that worse
			if worst_loss < val_loss:
				worst_loss = val_loss + 1
		break

	print("Total Bandits: {}".format(len(bandit_inst)))
	print("Worst Loss: {}".format(worst_loss))

	ush_inst = USH(300, bandit_inst, budget_type="time", max_number_of_round=3,
	               evaluation_rule=evaluation_rule, worst_loss=worst_loss)

	ush_inst.fit(x_train_fidelity, y_train_fidelity, random_state=rng)

	print("ush best bandit", ush_inst.get_key_of_best_bandit())

	best_val_loss = None
	best_ind = -1
	best_para = None
	for i in bandit_inst.keys():

		if best_val_loss is None or (bandit_inst[i].get_best_score() is not None
		                             and best_val_loss > bandit_inst[i].get_best_score()):
			best_val_loss = bandit_inst[i].get_best_score()
			best_ind = i
			best_para = bandit_inst[i].get_best_model_parameters()

	print("best ind: {}".format(best_ind))
	print("best val loss: ", best_val_loss)
	print("best bandit: {}".format(bandit_inst[best_ind].algorithm_model.get_model_name()))
	print("best parameter: {}".format(best_para))

	# refit
	bandit_inst[best_ind].algorithm_model.set_params(**best_para)
	bandit_inst[best_ind].algorithm_model.fit(x_train, y_train)
	y_pred = bandit_inst[best_ind].algorithm_model.predict(x_test)

	print("Final Test Score: {}".format(eval_performance(evaluation_rule, y_true=y_test, y_score=y_pred)))

	print("Total Actions: {}".format(sorted([x.get_num_of_actions() for x in bandit_inst.values()], reverse=True)))
