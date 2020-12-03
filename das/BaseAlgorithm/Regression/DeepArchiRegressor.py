import das
import copy
import logging
import numpy as np
from das.performance_evaluation import is_larger_better, eval_performance, judge_rule
from das.util.proba_utils import (from_probas_to_performance, infer_n_classes,
                                  aggregate_probas_to_proba, aggregate_probas_to_prediction)
from das.crossvalidate import cross_validate_transform
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.BaseAlgorithm.Regression.ArchiLayerRegressor import ArchiLayerRegressor

logger = logging.getLogger(das.logger_name)


class DeepArchiRegressor(BaseRegressor):

	def __init__(self,
	             base_layer: ArchiLayerRegressor,
	             max_layer=0,
	             early_stopping_rounds=1,
	             n_folds=3,
	             evaluation_rule='r2_score',
	             cross_validator=None,
	             n_classes=None,
	             e_id=None,
	             random_state=None):
		super(DeepArchiRegressor, self).__init__(e_id=e_id,
		                                         random_state=random_state, n_classes=n_classes)
		self.base_layer = base_layer
		self.max_layer = max_layer
		self.early_stopping_rounds = early_stopping_rounds
		assert self.max_layer > 0 or self.early_stopping_rounds > 0, ("max_layer or early_stopping rounds,"
		                                                              " at least one should be bigger than 0")
		self.n_folds = n_folds
		self.evaluation_rule = evaluation_rule
		self.cross_validator = cross_validator
		self.best_num_layers = None
		self.parallel_cv_transform = False
		# optimal training metric, train_performance or val_performance, generated after fit
		self.opt_train_metric = None
		self.layer_train_metrics = []
		self.layer_val_metrics = []

	@property
	def evaluation_rule_(self):
		return self.evaluation_rule

	def cross_validate_transform(self, **kwargs):
		if self.cross_validator is None:
			return cross_validate_transform(**kwargs)
		else:
			return self.cross_validator.cross_validate_transform(**kwargs)

	def _fit_predict(self, X, y, X_test=None, random_state=None, best_num_layers=None):
		"""
		fit_predict when best_num_layers is known.

		:param X:
		:param y:
		:param X_test:
		:param random_state:
		:param best_num_layers:
		:return:
		"""
		assert best_num_layers is not None, "Not fitted!!!"
		num_layer = 1
		n_train = X.shape[0]
		y_pred = np.zeros((n_train, 0), dtype=X.dtype)
		n_test, y_test_pred = None, None
		if X_test is not None:
			n_test = X_test.shape[0]
			y_test_pred = np.zeros((n_test, 0), dtype=X.dtype)
		while num_layer <= self.best_num_layers:
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			aug_X = np.hstack((X, y_pred))
			aug_X_test = None
			if X_test is not None:
				aug_X_test = np.hstack((X_test, y_test_pred))

			y_pred, y_test_pred = self.cross_validate_transform(
				model=model, X=aug_X, y=y, X_follow=aug_X_test, cv=self.n_folds,
				parallel=self.parallel_cv_transform, predict_method='predict_proba',
				task=self.task, random_state=random_state)
			del model
			num_layer += 1
		return y_pred, y_test_pred

	def fit_predict(self, X, y, X_test=None, random_state=None, best_num_layers=None, **kwargs):
		if random_state is None and self.random_state is not None:
			random_state = self.random_state
		# if False:  # best_num_layers is not None:
		# 	self.best_num_layers = best_num_layers
		# 	return self._fit_predict(X, y, X_test, random_state, best_num_layers)
		assert X_test is not None, "If no X_test, you should not invoke fit_predict, try fit."
		num_layer = 1
		self.classes_ = 1
		n_classes = 1
		assert isinstance(n_classes, int), "n_classes should be int type."
		n_train = X.shape[0]
		y_pred = np.zeros((n_train, 0), dtype=X.dtype)
		n_test, y_test_pred = None, None
		if X_test is not None:
			n_test = X_test.shape[0]
			y_test_pred = np.zeros((n_test, 0), dtype=X.dtype)
		self.layer_train_metrics = []
		ret_y_pred, ret_y_test_pred = None, None
		while True:
			if num_layer > self.max_layer > 0:
				break
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			aug_X = np.hstack((X, y_pred))
			aug_X_test = None
			if X_test is not None:
				aug_X_test = np.hstack((X_test, y_test_pred))

			y_pred, y_test_pred = self.cross_validate_transform(
				model=model, X=aug_X, y=y, X_follow=aug_X_test, cv=self.n_folds, parallel=self.parallel_cv_transform,
				predict_method='predict_proba', task=self.task, random_state=random_state, **kwargs)

			train_performance = from_probas_to_performance(y_pred, y, n_classes, task=self.task,
			                                               evaluation_rule=self.evaluation_rule)
			logger.info('Layer {}: train_{} = {}'.format(num_layer, self.evaluation_rule, train_performance))
			self.layer_train_metrics.append(train_performance)
			opt_layer_id = get_opt_layer_id(self.layer_train_metrics, is_larger_better(rule=self.evaluation_rule))
			if num_layer == 1 or opt_layer_id + 1 == num_layer:
				# This layer is the best layer, remember its output
				ret_y_pred, ret_y_test_pred = y_pred, y_test_pred
			self.opt_train_metric = self.layer_train_metrics[opt_layer_id]
			self.best_num_layers = opt_layer_id + 1
			if num_layer - opt_layer_id - 1 >= self.early_stopping_rounds + 3 > 0:
				break
			del model
			num_layer += 1
		return ret_y_pred, ret_y_test_pred

	def fit(self, X, y, X_val=None, y_val=None, random_state=None, **kwargs):
		"""
		If X_val is passed in, then early stopping by val_performance criteria.
		Otherwise (no X_val), we early stop the deep architecture by train_performance criteria.

		:param X:
		:param y:
		:param X_val:
		:param y_val:
		:param random_state:
		:return:
		"""
		if random_state is None and self.random_state is not None:
			random_state = self.random_state
		num_layer = 1
		self.classes_ = 1
		n_classes = 1
		assert isinstance(n_classes, int) and n_classes is not None,\
			"n_classes should be int type and not None."
		n_train = X.shape[0]
		y_pred = np.zeros((n_train, 0), dtype=X.dtype)
		n_val, y_val_pred, final_val_proba = None, None, None
		if X_val is not None:
			n_val = X_val.shape[0]
			y_val_pred = np.zeros((n_val, 0), dtype=X.dtype)
		self.layer_train_metrics = []
		self.layer_val_metrics = []
		while True:
			if num_layer > self.max_layer > 0:
				break
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			aug_X = np.hstack((X, y_pred))
			aug_X_val = None
			if X_val is not None:
				aug_X_val = np.hstack((X_val, y_val_pred))

			y_pred, y_val_pred = self.cross_validate_transform(
				model=model, X=aug_X, y=y, X_follow=aug_X_val, cv=self.n_folds,
				predict_method='predict_proba', parallel=self.parallel_cv_transform,
				task=self.task, random_state=random_state, **kwargs)

			train_performance = from_probas_to_performance(y_pred, y, n_classes, task=self.task,
			                                               evaluation_rule=self.evaluation_rule)
			logger.info('Layer {}: train_{} = {}'.format(num_layer, self.evaluation_rule, train_performance))
			self.layer_train_metrics.append(train_performance)
			if X_val is not None:
				val_performance = from_probas_to_performance(y_val_pred, y_val, n_classes, task=self.task,
				                                             evaluation_rule=self.evaluation_rule)
				logger.info('Layer {}: val_{} = {}'.format(num_layer, self.evaluation_rule, val_performance))
				self.layer_val_metrics.append(val_performance)
			if 'stop_by_val' in kwargs and kwargs['stop_by_val'] is True:
				opt_layer_id = get_opt_layer_id(self.layer_val_metrics, is_larger_better(self.evaluation_rule))
				self.opt_train_metric = self.layer_val_metrics[opt_layer_id]
			else:
				opt_layer_id = get_opt_layer_id(self.layer_train_metrics, is_larger_better(self.evaluation_rule))
				self.opt_train_metric = self.layer_train_metrics[opt_layer_id]
			self.best_num_layers = opt_layer_id + 1
			if num_layer - opt_layer_id - 1 >= self.early_stopping_rounds > 0:
				break
			if self.evaluation_rule == 'r2_score' and self.opt_train_metric < 0:  # if r2_score is negative, early stop
				break
			del model
			num_layer += 1

		return self

	def get_configuration_space(self):
		return self.base_layer.get_configuration_space()

	def get_config_space(self):
		return self.base_layer.get_config_space()

	def get_model_name(self, concise=False):
		return "DeepArchiRegressor({})".format(self.base_layer.get_model_name(concise=concise))

	def get_model(self):
		raise NotImplementedError("Now we cannot provide model to you")

	def get_model_type(self):
		return DeepArchiRegressor

	def get_params(self, deep=True, only_cfg=True):
		return self.base_layer.get_params(deep=deep, only_cfg=only_cfg)

	def set_configuration_space(self, ps=None):
		self.base_layer.set_configuration_space(ps=ps)

	def set_params(self, **kwargs):
		self.base_layer.set_params(**kwargs)


def get_opt_layer_id(acc_list, larger_better=True):
	""" Return layer id with max accuracy on training data """
	if larger_better:
		opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
	else:
		opt_layer_id = np.argsort(np.asarray(acc_list), kind='mergesort')[0]
	return opt_layer_id


if __name__ == '__main__':
	from das.BaseAlgorithm.Regression.ExtraTreesRegressor import ExtraTreesRegressor
	from das.BaseAlgorithm.Regression.RandomForestRegressor import RandomForestRegressor
	from das.BaseAlgorithm.Regression.ArchiBlockRegressor import HorizontalBlockRegressor

	hbc = HorizontalBlockRegressor(4, RandomForestRegressor, model_params={'n_estimators': 50})
	hbc2 = HorizontalBlockRegressor(4, ExtraTreesRegressor, model_params={'n_estimators': 50})
	alc = ArchiLayerRegressor(2, [("hbc", hbc), ("EXT", hbc2)], e_id=0)
	print(alc.get_model_name())
	import ray
	ray.init()
	# hbc.get_configuration_space().show_space_names()
	from benchmarks.data.mg.load_mg import load_mg
	# logger.setLevel('DEBUG')
	x_train, x_test, y_train, y_test = load_mg()
	# alc.fit(x_train, y_train)
	# ans = alc.score(x_test, y_test, evaluation_rule='accuracy_score')
	# print(ans)
	import time
	start_time = time.time()
	print("Now =================================")
	dac = DeepArchiRegressor(base_layer=alc, max_layer=0, early_stopping_rounds=1, n_folds=3,
	                         evaluation_rule='r2_score', n_classes=1, e_id='dac', random_state=0)
	dac.fit(x_train, y_train, distribute=False)
	print("opt metric, best_num_layers ", dac.opt_train_metric, dac.best_num_layers)
	print("dac.layer_train_metrics ", dac.layer_train_metrics)
	print("dac.layer_val_metrics ", dac.layer_val_metrics)
	print("TimeCost: {}".format(time.time()-start_time))

