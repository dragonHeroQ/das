import sys
sys.path.append("../")
sys.path.append("../../")
import das
import importlib
import os.path as osp
from math import log, ceil
# from das.ArchitectureSearch.Optimizer.BayesianOptimizer import BayesianOptimizer
from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
from das.ArchitectureSearch.SearchFramework.hyperband.HyperBandIteration import SuccessiveHalvingIteration
from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator
from das.ArchitectureSearch.SamplingStrategy import SizeSamplingStrategy

util = importlib.import_module('util')
extract_specification = getattr(util, 'extract_specification')
experiment_summary = getattr(util, 'experiment_summary')
get_total_MBSize = getattr(util, 'get_total_MBSize')

search_algo = osp.relpath(__file__)[:-3]
return_dict = extract_specification(sys.argv, search_algo=search_algo)

experiment_summary(return_dict)

x_train, x_test, y_train, y_test = return_dict['dataset_tuple']
rng = return_dict['rng']
total_budget = return_dict['total_budget']
budget_type = return_dict['budget_type']
n_classes = return_dict['n_classes']
evaluation_rule = return_dict['evaluation_rule']
lcv_f_name = return_dict['lcv_f_name']
run_timelimit = 240.0
if 'TL' in return_dict:
	run_timelimit = return_dict['TL']

if 'log' in return_dict:
	import logging
	logger = logging.getLogger(das.logger_name)
	logger.setLevel(return_dict['log'])

sampling_strategy = None
if 'sample' in return_dict:
	size_sample = float(return_dict['sample'])
	sampling_strategy = SizeSamplingStrategy(size_sample)

is_debugging = False
if 'debug' in return_dict:
	is_debugging = True if return_dict['debug'] == 'True' else False

learning_tool = DeepArchiLearningTool(n_block=2, n_classes=n_classes, evaluation_rule=evaluation_rule)
evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule=evaluation_rule)
optimizer = RandomSearchOptimizer(parameter_space=learning_tool.get_parameter_space())

num_bandits = 81
if 'nbandit' in return_dict:
	num_bandits = int(return_dict['nbandit'])

eta = 3
if 'eta' in return_dict:
	eta = int(return_dict['eta'])

max_num_stages = int(ceil(log(num_bandits, eta))) + 1
print("Max Num Stages: {}".format(max_num_stages))

all_bandits = dict([(i + 1,
                     learning_tool.create_learning_tool(**optimizer.get_next_config()))
                    for i in range(num_bandits)])

sh_iter = SuccessiveHalvingIteration(eta=eta, iteration_budget=total_budget, budget_type=budget_type,
                                     all_bandits=all_bandits, evaluator=evaluator, sampling_strategy=sampling_strategy,
                                     stage_time_controller=None, max_num_stages=max_num_stages,
                                     evaluation_rule=evaluation_rule, per_run_timelimit=run_timelimit,
                                     random_state=rng)

sh_iter.fit(x_train, y_train, confidence_screening=True, debug=is_debugging)
train_score, test_score = sh_iter.refit_and_score(x_train, y_train, x_test, y_test,
                                                  run_time_limit=run_timelimit * 10 * max(1, get_total_MBSize(x_train, x_test)))
print("Final Train Score={}, Test Score={}".format(train_score, test_score))
if test_score is None:
	test_score = 0
evaluator.save_learning_curve(f_name="{}({:.6f}).lcv".format(lcv_f_name[:-4], test_score), base_dir="./lcvs")
