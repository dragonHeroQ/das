import sys
sys.path.append("../")
sys.path.append("../../")
import das
import ray
import importlib
import os.path as osp
from das.ArchitectureSearch.SearchFramework.evolutionary.AgingEvolution import AgingEvolution
from das.ArchitectureSearch.SearchFramework.evolutionary.DisAgingEvolution import DisAgingEvolution
from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator
from das.ArchitectureSearch.Evaluator.DisEvaluatorProxy import DisEvaluatorProxy
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

P = 100
if 'P' in return_dict:
	P = int(return_dict['P'])

S = 25
if 'S' in return_dict:
	S = int(return_dict['S'])

neval = 2
if 'neval' in return_dict:
	neval = int(return_dict['neval'])

mtsn = 4
if 'mtsn' in return_dict:
	mtsn = int(return_dict['mtsn'])

ray.init()

learning_tool = DeepArchiLearningTool(n_block=2, n_classes=n_classes, evaluation_rule=evaluation_rule)
# evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule=evaluation_rule)

evaluator = DisEvaluatorProxy(num_evaluators=neval,
                              evaluator_params={'n_folds': 3, 'evaluation_rule': evaluation_rule},
                              max_task_at_single_evaluator=1,
                              max_task_at_single_node=mtsn,
                              n_folds=3, evaluation_rule=evaluation_rule)

DAE = DisAgingEvolution(evaluator=evaluator, learning_tool=learning_tool, sampling_strategy=sampling_strategy,
                        total_budget=total_budget, budget_type=budget_type, optimizer_class=RandomSearchOptimizer,
                        per_run_timelimit=run_timelimit, evaluation_rule=evaluation_rule,
                        P=P, S=S, random_state=rng)

DAE.fit(x_train, y_train, confidence_screening=True, debug=is_debugging)
print("get_total_MBSize(x_train, x_test) = {}".format(get_total_MBSize(x_train, x_test)))
train_score, test_score = DAE.refit_and_score(x_train, y_train, x_test, y_test,
                                              run_time_limit=run_timelimit * 10 * get_total_MBSize(x_train, x_test))
print("Final Train Score={}, Test Score={}".format(train_score, test_score))
if test_score is None:
	test_score = 0
evaluator.save_learning_curve(f_name="{}({:.6f}).lcv".format(lcv_f_name[:-4], test_score), base_dir="./lcvs")