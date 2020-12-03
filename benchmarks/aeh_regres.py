import sys
sys.path.append("../")
sys.path.append("../../")
import das
import importlib
import os.path as osp
from das.ArchitectureSearch.SearchFramework.evolutionary.AgingEvolution import AgingEvolution
from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator
from das.ArchitectureSearch.SamplingStrategy import SizeSamplingStrategy
from das.BaseAlgorithm.algorithm_space import get_all_regression_algorithm_keys

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

P = 100
if 'P' in return_dict:
	P = int(return_dict['P'])

S = 25
if 'S' in return_dict:
	S = int(return_dict['S'])

# import ray
# ray.init(redis_address="192.168.100.35:6379")
algo_space = get_all_regression_algorithm_keys()
# algo_space.remove('LGBClassifier')
# algo_space.remove('XGBClassifier')
learning_tool = DeepArchiLearningTool(n_block=2, n_classes=n_classes,
                                      evaluation_rule=evaluation_rule,
                                      algo_space=algo_space)
evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule=evaluation_rule, redis_address="192.168.100.35:6379")
AE = AgingEvolution(evaluator=evaluator, learning_tool=learning_tool, sampling_strategy=sampling_strategy,
                    total_budget=total_budget, budget_type=budget_type, optimizer_class=RandomSearchOptimizer,
                    per_run_timelimit=run_timelimit, evaluation_rule=evaluation_rule,
                    P=P, S=S, random_state=rng)

AE.fit(x_train, y_train, distribute=0)
print("get_total_MBSize(x_train, x_test) = {}".format(get_total_MBSize(x_train, x_test)))
train_score, test_score = AE.refit_and_score(x_train, y_train, x_test, y_test,
                                             run_time_limit=run_timelimit * 10 * max(1, get_total_MBSize(x_train, x_test)))
print("Final Train Score={}, Test Score={}".format(train_score, test_score))
if test_score is None:
	test_score = 0
evaluator.save_learning_curve(f_name="{}({:.6f}).lcv".format(lcv_f_name[:-4], test_score), base_dir="./lcvs")
