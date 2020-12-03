import sys

sys.path.append("../")
sys.path.append("../../")
import das
import importlib
import os.path as osp
from das.BaseAlgorithm.Regression.ArchiLayerRegressor import ArchiLayerRegressor
from das.BaseAlgorithm.Regression.DeepArchiRegressor import DeepArchiRegressor

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

from das.BaseAlgorithm.Regression.ExtraTreesRegressor import ExtraTreesRegressor
from das.BaseAlgorithm.Regression.RandomForestRegressor import RandomForestRegressor
from das.BaseAlgorithm.Regression.ArchiBlockRegressor import HorizontalBlockRegressor

hbc1 = HorizontalBlockRegressor(4, RandomForestRegressor,
                                model_params={'n_estimators': 500, 'max_depth': 100})
hbc2 = HorizontalBlockRegressor(4, ExtraTreesRegressor,
                                model_params={'n_estimators': 500, 'max_depth': 100, 'max_features': 1})
alc = ArchiLayerRegressor(2, [("RF", hbc1), ("ERF", hbc2)], e_id=0, random_state=rng)
print(alc.get_model_name())

import ray

ray.init()

import time

start_time = time.time()
print("Now =================================")
dac = DeepArchiRegressor(base_layer=alc, max_layer=0, early_stopping_rounds=4, n_folds=3,
                         evaluation_rule=evaluation_rule, n_classes=n_classes,
                         e_id='dac_{}'.format(rng), random_state=rng)
dac.fit(x_train, y_train, x_test, y_test, distribute=1)
print("opt metric, best_num_layers ", dac.opt_train_metric,
      dac.layer_val_metrics[dac.best_num_layers - 1], dac.best_num_layers)
print("dac.layer_train_metrics ", dac.layer_train_metrics)
print("dac.layer_val_metrics ", dac.layer_val_metrics)
print("Time Cost = {}".format(time.time() - start_time))

learning_curve = {time.time() - start_time: {'val_{}'.format(evaluation_rule): dac.opt_train_metric}}

f_name = "{}({:.6f}).lcv".format(lcv_f_name[:-4], dac.layer_val_metrics[dac.best_num_layers - 1])
base_dir = "./lcvs"

import os
import pickle

if not osp.exists(base_dir):
	os.makedirs(base_dir)

print("Learning Curve Saved in {}".format(osp.join(base_dir, f_name)))
pickle.dump(learning_curve, open(osp.join(base_dir, f_name), 'wb'))
