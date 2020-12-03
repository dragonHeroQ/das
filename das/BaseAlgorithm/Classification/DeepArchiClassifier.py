import das
import copy
import logging
import numpy as np
from das.performance_evaluation import is_larger_better, eval_performance, judge_rule
from das.util.proba_utils import (from_probas_to_performance, infer_n_classes,
                                  aggregate_probas_to_proba, aggregate_probas_to_prediction)
from das.crossvalidate import cross_validate_transform
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.BaseAlgorithm.Classification.ArchiLayerClassifier import ArchiLayerClassifier

logger = logging.getLogger(das.logger_name)


class DeepArchiClassifier(BaseClassifier):

	def __init__(self,
	             base_layer: ArchiLayerClassifier,
	             max_layer=0,
	             early_stopping_rounds=1,
	             n_folds=3,
	             evaluation_rule='accuracy_score',
	             cross_validator=None,
	             n_classes=None,
	             e_id=None,
	             random_state=None):
		super(DeepArchiClassifier, self).__init__(e_id=e_id,
		                                          random_state=random_state, n_classes=n_classes)
		self.classes_ = []
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

	def _fit_predict(self, X, y, X_test=None, random_state=None, best_num_layers=None):
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
		self.handle_y_mapping(y)
		# if False:  # best_num_layers is not None:
		# 	self.best_num_layers = best_num_layers
		# 	return self._fit_predict(X, y, X_test, random_state, best_num_layers)
		assert X_test is not None, "If no X_test, you should not invoke fit_predict, try fit."
		num_layer = 1
		if self.n_classes_ is None:
			self.n_classes_ = infer_n_classes(y=y, evaluation_rule=self.evaluation_rule)
		n_classes = self.n_classes_
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
			logger.debug("Layer {} Started!".format(num_layer))
			if num_layer > self.max_layer > 0:
				break
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			aug_X = np.hstack((X, y_pred))
			aug_X_test = None
			if X_test is not None:
				aug_X_test = np.hstack((X_test, y_test_pred))

			y_pred, y_test_pred = self.cross_validate_transform(
				model=model, X=aug_X, y=y, X_follow=aug_X_test, cv=self.n_folds,
				parallel=self.parallel_cv_transform, predict_method='predict_proba',
				task=self.task, random_state=random_state, **kwargs)
			logger.debug("y_pred and y_test_pred, shape = ", y_pred.shape, y_test_pred.shape)
			train_performance = from_probas_to_performance(y_pred, y, n_classes,
			                                               evaluation_rule=self.evaluation_rule,
			                                               classes_=self.classes_)
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

	def cross_validate_transform(self, **kwargs):
		if self.cross_validator is None:
			return cross_validate_transform(**kwargs)
		else:
			return self.cross_validator.cross_validate_transform(**kwargs)

	def fit_cs(self, X, y, X_val=None, y_val=None, random_state=None, **kwargs):
		# print("Come on!")
		num_layer = 1
		if self.n_classes_ is None:
			self.n_classes_ = infer_n_classes(y=y, evaluation_rule=self.evaluation_rule)
		n_classes = self.n_classes_
		assert isinstance(n_classes, int) and n_classes is not None,\
			"n_classes should be int type and not None."
		n_train = X.shape[0]
		Y_pred = np.zeros((n_train, 0))
		n_val, y_val_pred, final_val_proba = None, None, None
		if X_val is not None:
			n_val = X_val.shape[0]
			y_val_pred = np.zeros((n_val, 0), dtype=X.dtype)
		self.layer_train_metrics = []
		self.layer_val_metrics = []
		mask = [0 for _ in range(n_train)]
		a = 1 / 3.0
		while True:
			if num_layer > self.max_layer > 0:
				break
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			mapping = dict()
			cur_n_train = len(mask) - sum(mask)
			if cur_n_train < self.n_folds:
				break
			# print("Cur_n_train = {}".format(cur_n_train))
			new_X = np.zeros((cur_n_train, X.shape[1]), dtype=X.dtype)
			new_y = np.zeros((cur_n_train, ), dtype=y.dtype)
			new_pred = np.zeros((cur_n_train, Y_pred.shape[1]))
			k = 0
			for i in range(len(mask)):
				if mask[i] == 0:
					new_X[k, :] = X[i, :]
					new_y[k] = y[i]
					new_pred[k, :] = Y_pred[i, :]
					mapping[k] = i
					k += 1
			if num_layer > 1:
				aug_X = np.hstack((new_X, new_pred))
			else:
				aug_X = new_X
			aug_X_val = None
			if X_val is not None:
				aug_X_val = np.hstack((X_val, y_val_pred))

			# print("aug_X.shape = {}, new_y.shape = {}".format(aug_X.shape, new_y.shape))
			y_pred, y_val_pred = self.cross_validate_transform(
				model=model, X=aug_X, y=new_y, X_follow=aug_X_val, cv=self.n_folds,
				predict_method='predict_proba', parallel=self.parallel_cv_transform,
				task=self.task, random_state=random_state, **kwargs)

			# y_pred = aggregate_probas_to_proba(y_pred, n_classes)
			# print("y_pred = {}".format(y_pred))
			if num_layer == 1:
				Y_pred = np.hstack((Y_pred, y_pred))
			else:
				for i in range(y_pred.shape[0]):
					Y_pred[mapping[i], :] = y_pred[i, :]

			train_performance = from_probas_to_performance(Y_pred, y, n_classes,
			                                               evaluation_rule=self.evaluation_rule,
			                                               classes_=self.classes_)

			if num_layer == 1:
				if train_performance > 0.9:
					a = 1 / 10.0
				else:
					a = 1 / 3.0

			eta_t = get_eta_t(a=a, acc_t=train_performance, n_classes=n_classes, Y_pred=Y_pred, y=y,
			                  evaluation_rule=self.evaluation_rule, classes_=self.classes_)
			# print("eta_t = {}".format(eta_t))

			aggregated_y_pred = aggregate_probas_to_proba(y_pred, n_classes)
			for i in range(y_pred.shape[0]):
				if confidence(aggregated_y_pred[i, :]) >= eta_t:
					mask[mapping[i]] = 1

			logger.info('Layer {}: train_{} = {}'.format(num_layer, self.evaluation_rule, train_performance))
			self.layer_train_metrics.append(train_performance)
			if X_val is not None:
				val_performance = from_probas_to_performance(y_val_pred, y_val, n_classes,
				                                             evaluation_rule=self.evaluation_rule,
				                                             classes_=self.classes_)
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
			del model
			num_layer += 1

		return self

	def handle_y_mapping(self, y):
		self.classes_ = []
		if self.task == 'classification':
			uni_y = np.unique(y)
			for cls in uni_y:
				self.classes_.append(cls)

	def fit(self, X, y, X_val=None, y_val=None, random_state=None, **kwargs):
		"""
		If X_val is passed in, then early stopping by val_performance criteria.
		Otherwise (no X_val), we early stop the deep architecture by train_performance criteria.

		confidence_screening, default False: whether to open confidence_screening to boosting training speed.
		distribute, default 0: distribute computing level. 0 means sequential, 1 means block-level parallelism and 2
								means element-level parallelism.

		:param X:
		:param y:
		:param X_val:
		:param y_val:
		:param random_state:
		:return:
		"""
		if random_state is None and self.random_state is not None:
			random_state = self.random_state
		self.handle_y_mapping(y)
		if 'confidence_screening' in kwargs and kwargs['confidence_screening'] is True:
			kwargs.pop('confidence_screening')
			return self.fit_cs(X, y, X_val, y_val, random_state, **kwargs)
		num_layer = 1
		if self.n_classes_ is None:
			self.n_classes_ = infer_n_classes(y=y, evaluation_rule=self.evaluation_rule)
		n_classes = self.n_classes_
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

			train_performance = from_probas_to_performance(y_pred, y, n_classes,
			                                               evaluation_rule=self.evaluation_rule,
			                                               classes_=self.classes_)
			logger.info('Layer {}: train_{} = {}'.format(num_layer, self.evaluation_rule, train_performance))
			self.layer_train_metrics.append(train_performance)
			if X_val is not None:
				val_performance = from_probas_to_performance(y_val_pred, y_val, n_classes,
				                                             evaluation_rule=self.evaluation_rule,
				                                             classes_=self.classes_)
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
			del model
			num_layer += 1

		return self

	def fit_ray(self, Xid, y, random_state=None, **kwargs):
		if random_state is None and self.random_state is not None:
			random_state = self.random_state
		self.handle_y_mapping(y)
		num_layer = 1
		if self.n_classes_ is None:
			self.n_classes_ = infer_n_classes(y=y, evaluation_rule=self.evaluation_rule)
		n_classes = self.n_classes_
		assert isinstance(n_classes, int) and n_classes is not None, "n_classes should be int type and not None."
		n_train = y.shape[0]
		y_pred = np.zeros((n_train, 0))
		self.layer_train_metrics = []
		assert 'distribute' in kwargs and kwargs['distribute'] > 0, "fit_ray(..) requires: distribute > 0"
		distribute = kwargs['distribute']
		kwargs.pop('distribute')
		while True:
			if num_layer > self.max_layer > 0:
				break
			model = copy.deepcopy(self.base_layer)
			model.set_e_id("{}#L{}".format(self.e_id, num_layer))
			logger.info("Layer {} start to train ...".format(num_layer))
			y_pred = model.fit_predict_kfold_ray(Xid, y, y_pred, cv=self.n_folds,
			                                     predict_method='predict_proba',
			                                     task=self.task, random_state=random_state,
			                                     distribute=distribute, **kwargs)
			assert y_pred is not None, "Exception Occurred, y_pred in layer {} is None".format(num_layer)
			train_performance = from_probas_to_performance(y_pred, y, n_classes,
			                                               evaluation_rule=self.evaluation_rule,
			                                               classes_=self.classes_)
			logger.info('Layer {}: train_{} = {}'.format(num_layer, self.evaluation_rule, train_performance))
			self.layer_train_metrics.append(train_performance)
			opt_layer_id = get_opt_layer_id(self.layer_train_metrics, is_larger_better(self.evaluation_rule))
			self.opt_train_metric = self.layer_train_metrics[opt_layer_id]
			self.best_num_layers = opt_layer_id + 1
			if num_layer - opt_layer_id - 1 >= self.early_stopping_rounds > 0:
				break
			del model
			num_layer += 1

		return self

	def get_configuration_space(self):
		return self.base_layer.get_configuration_space()

	def get_config_space(self):
		return self.base_layer.get_config_space()

	def get_model_name(self, concise=False):
		return "DeepArchiClassifier({})".format(self.base_layer.get_model_name(concise=concise))

	def get_model(self):
		raise NotImplementedError("Now we cannot provide model to you")

	def get_model_type(self):
		return DeepArchiClassifier

	def get_params(self, deep=True, only_cfg=True):
		return self.base_layer.get_params(deep=deep, only_cfg=only_cfg)

	def set_configuration_space(self, ps=None):
		self.base_layer.set_configuration_space(ps=ps)

	def set_params(self, **kwargs):
		self.base_layer.set_params(**kwargs)


def confidence(np_array):
	sum_value = np.sum(np_array)
	np_array /= sum_value
	return np.max(np_array)


def get_eta_t(a, acc_t, n_classes, Y_pred, y, evaluation_rule, classes_):
	aggregated_Y_pred = aggregate_probas_to_proba(Y_pred, n_classes)
	confidences = []
	for i in range(Y_pred.shape[0]):
		confidences.append((i, confidence(aggregated_Y_pred[i, :])))
	# n_classes = Y_pred.shape[1]
	sorted_conf = sorted(confidences, key=lambda x: x[1], reverse=True)
	start = 0
	low = 0
	high = len(sorted_conf)-1
	while low <= high:
		mid = (low + high) // 2
		# y_pred = []
		# y_true = []
		indexes = list(map(lambda x: sorted_conf[x][0], range(start, mid+1)))
		y_pred = aggregate_probas_to_prediction(aggregated_Y_pred[indexes], n_classes)
		y_true = y[indexes]
		# print("[get_eta_t] y_pred.shape = {}, y_true.shape = {}".format(y_pred.shape, y_true.shape))
		# for i in range(low, mid+1):
		# 	y_pred.append(aggregate_probas_to_prediction(Y_pred[sorted_conf[i][0]], n_classes))
		# 	y_true.append(y[sorted_conf[i][0]])
		if judge_rule(evaluation_rule) == 'classification':
			y_pred = np.array(list(map(lambda x: classes_[x], y_pred)))
		acc = eval_performance(rule=evaluation_rule, y_true=y_true, y_score=y_pred)
		# print("acc = {}, acc_t = {}".format(acc, acc_t))
		if acc > 1 - a + a * acc_t:
			low = mid + 1
		else:
			high = mid - 1
	return sorted_conf[low][1]


def get_opt_layer_id(acc_list, larger_better=True):
	""" Return layer id with max accuracy on training data """
	if larger_better:
		opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
	else:
		opt_layer_id = np.argsort(np.asarray(acc_list), kind='mergesort')[0]
	return opt_layer_id


if __name__ == '__main__':
	from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
	from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
	from das.BaseAlgorithm.Classification.GBDT import GBDTClassifier
	from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier, VerticalBlockClassifier

	hbc = HorizontalBlockClassifier(4, RandomForestClassifier, model_params={'n_estimators': 70})
	hbc2 = HorizontalBlockClassifier(4, ExtraTreesClassifier, model_params={'n_estimators': 70})
	alc = ArchiLayerClassifier(2, [("hbc", hbc), ('hbc2', hbc2)], e_id=0)
	print(alc.get_model_name())
	# hbc.get_configuration_space().show_space_names()
	from benchmarks.data.letter.load_letter import load_letter
	from benchmarks.data.digits.load_digits import load_digits
	from benchmarks.data.dexter.load_dexter import load_dexter
	from benchmarks.data.yeast.load_yeast import load_yeast
	from benchmarks.data.adult.load_adult import load_adult
	# logger.setLevel('DEBUG')
	x_train, x_test, y_train, y_test = load_yeast()
	# alc.fit(x_train, y_train)
	# ans = alc.score(x_test, y_test, evaluation_rule='accuracy_score')
	# print(ans)
	import ray
	ray.init()
	import time
	start_time = time.time()
	print("Now =================================")
	dac = DeepArchiClassifier(base_layer=alc, max_layer=1, early_stopping_rounds=1, n_folds=3,
	                          evaluation_rule='accuracy_score', e_id='dac', random_state=0)
	dac.fit(x_train, y_train, x_test, y_test, confidence_screening=True, distribute=1)

	hbc = HorizontalBlockClassifier(4, GBDTClassifier, model_params={'n_estimators': 70})
	hbc2 = HorizontalBlockClassifier(4, ExtraTreesClassifier, model_params={'n_estimators': 70})
	alc = ArchiLayerClassifier(2, [("hbc", hbc), ('hbc2', hbc2)], e_id=0)
	dac = DeepArchiClassifier(base_layer=alc, max_layer=0, early_stopping_rounds=1, n_folds=3,
	                          evaluation_rule='accuracy_score', e_id='dac', random_state=0)
	dac.fit(x_train, y_train, x_test, y_test, confidence_screening=True, distribute=1)

	print("opt metric, best_num_layers ", dac.opt_train_metric, dac.best_num_layers)
	print("dac.layer_train_metrics ", dac.layer_train_metrics)
	print("dac.layer_val_metrics ", dac.layer_val_metrics)
	print("Time Cost = {}".format(time.time()-start_time))

	# from sklearn.model_selection import cross_val_score
	#
	# rf = RandomForestClassifier(n_estimators=100)
	# scores = cross_val_score(rf, x_train, y_train, cv=3)
	# print(scores)
	# rf.fit(x_train, y_train)
	# y_pred_test = rf.predict(x_test)
	# print(y_pred_test)
