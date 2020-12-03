import sys
sys.path.append("../")
sys.path.append("../../")
import das
import importlib
import os.path as osp

util = importlib.import_module('util')
extract_specification = getattr(util, 'extract_specification')
experiment_summary = getattr(util, 'experiment_summary')

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

# ===

from das.performance_evaluation import eval_performance
from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)

import time
start_time = time.time()

rf.fit(x_train, y_train)
y_pred_test = rf.predict(x_test)
y_pred = rf.predict(x_train)

train_acc = eval_performance(rule=evaluation_rule, y_true=y_train, y_score=y_pred)
test_acc = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_pred_test)

print("opt metric", train_acc, test_acc)
print("Time Cost = {}".format(time.time()-start_time))
