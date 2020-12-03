import sys
import time
import pickle
import warnings
import importlib
sys.path.append("../")
sys.path.append("../..")
warnings.filterwarnings("ignore")
from automl.HyperparameterOptimizer.USH import USH, CompModelFilter, Bandit
from automl.compmodel import CompositeModel
from automl.performance_evaluation import *
from automl.get_algorithm import (get_algorithm_by_key,
                                  get_all_regression_algorithm_keys)

if len(sys.argv) > 1:
	data_name = sys.argv[1]
else:
	data_name = 'mg'
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
	time_budget = 1200
print("Total Time Budget = {}".format(time_budget))

if __name__ == "__main__":

	start_time = time.time()
	evaluation_rule = 'mean_squared_error'
	pickle_name = "{}_mses_{}".format(data_name, rng)

	comp_model_filter = CompModelFilter(total_budget=time_budget, budget_type='time', minimum_run_timelimit=10.0,
	                                    per_run_timelimit=240.0, pickle_name=pickle_name, time_ref=start_time,
	                                    evaluation_rule=evaluation_rule, worst_loss=np.inf, random_state=rng)

	model_space = comp_model_filter.construct_composite_model_space()
	# model_space[(idx, m1, m2, cat_type)] = 1.0

	bandit_inst = {}
	idx = 0
	worst_loss = np.inf
	for (m_idx, cm1, cm2, cat_type) in model_space:
		default_comp_model = CompositeModel([(cm1, get_algorithm_by_key(cm1, random_state=rng)),
		                                     (cm2, get_algorithm_by_key(cm2, random_state=42 + rng)),
		                                     cat_type])
		bandit_inst[idx] = Bandit(algorithm_model=default_comp_model)
		idx += 1

	print("Total Bandits: {}".format(len(bandit_inst)))
	print("Worst Loss: {}".format(worst_loss))

	ush_inst = USH(7200, bandit_inst, budget_type="time", max_number_of_round=4, per_run_timelimit=240.0,
	               evaluation_rule=evaluation_rule, time_ref=start_time, worst_loss=worst_loss)

	ush_inst.fit(x_train, y_train, fidelity_mb=0.5, random_state=rng)
	ush_inst.refit(x_train, y_train)
	ush_inst.score(x_test, y_test)
	print("TOTAL TimeCost: {}".format(time.time()-start_time))
	learning_curve = {}
	learning_curve.update(comp_model_filter.get_learning_curve())
	learning_curve.update(ush_inst.get_learning_curve())
	pickle.dump(learning_curve, open("pure_round_{}.lcv".format(pickle_name), 'wb'))
