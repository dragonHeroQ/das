import os
import das
import time
import copy
import logging
import traceback
import numpy as np
import os.path as osp
import multiprocessing
from das.util.common_utils import kill_tree, search_newest_from_dir
from das.performance_evaluation import score_to_loss, loss_to_score, initial_worst_loss
from das.BaseAlgorithm.Classification.DeepArchiClassifier import DeepArchiClassifier
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool

logger = logging.getLogger(das.logger_name)


class BaseEvaluator(object):

	def __init__(self,
	             validation_strategy='cv',
	             validation_strategy_args=3,
	             evaluation_rule=None,
	             worst_loss=None,
	             time_ref=None,
	             **kwargs):
		self.validation_strategy = validation_strategy
		self.validation_strategy_args = validation_strategy_args
		if self.validation_strategy == 'cv':
			self.n_folds = int(self.validation_strategy_args)
		self.evaluation_rule = evaluation_rule
		self.time_ref = time_ref or time.time()
		self.worst_loss = worst_loss or initial_worst_loss(rule=self.evaluation_rule)
		for key in kwargs:
			setattr(self, key, kwargs[key])

		# running attrs
		self.learning_curve = {}  # {time:performance}
		self.fit_predict_reward = None
		self.cur_best_val_score = -np.inf
		self.evaluation_count = 0

	def set_evaluation_rule(self, evaluation_rule=None):
		self.evaluation_rule = evaluation_rule

	def evaluate(self, learning_tool, X, y, run_time_limit=240.0, random_state=None, **kwargs):
		raise NotImplementedError

	@staticmethod
	def _fit_predict(learning_estimator, X, y, X_test, return_dict, random_state=None, **kwargs):
		y_pred, y_test_pred = None, None
		try:
			y_pred, y_test_pred = learning_estimator.fit_predict(X, y, X_test, random_state, **kwargs)
		except Exception as e:
			print(e)
			print(traceback.format_exc())
			return_dict['exception'] = str(e)
		finally:
			return_dict['y_pred'] = y_pred
			return_dict['y_test_pred'] = y_test_pred

	def fit_predict(self, learning_tool: BaseLearningTool, X, y, X_test,
	                run_time_limit=360.0, random_state=None, **kwargs):
		learning_estimator = learning_tool.learning_estimator
		if self.validation_strategy == 'cv':
			learning_estimator.n_folds = self.n_folds
		y_pred, y_test_pred = None, None
		start_time = time.time()
		error_message = None
		logger.info("fit_predict TimeLimit: {}".format(run_time_limit))
		try:
			mgr = multiprocessing.Manager()
			return_dict = mgr.dict()
			p = multiprocessing.Process(target=self._fit_predict,
			                            args=(learning_estimator, X, y, X_test, return_dict, random_state),
			                            kwargs=kwargs)
			p.start()
			p.join(run_time_limit)

			if p.is_alive():
				p.terminate()
				kill_tree(p.pid)

			if 'y_pred' in return_dict and return_dict['y_pred'] is not None:
				y_pred = return_dict['y_pred']
				y_test_pred = return_dict['y_test_pred']
				if 'exception' in return_dict:
					error_message = return_dict['exception']
			else:
				error_message = 'May be TimeOut'

		except Exception as e:
			logger.warning("Exceptions Occurred")
			error_message = str(e)
			print(e)
		finally:
			time_cost = time.time() - start_time
			logger.info("TimeCost: {}, Exceptions: {}".format(time_cost, error_message))
			return y_pred, y_test_pred

	def record_learning_curve(self, reward, learning_tool=None, model_params=None):
		rule = self.evaluation_rule
		self.learning_curve[time.time()-self.time_ref] = {
			'val_{}'.format(rule): reward['val_{}'.format(rule)],
			'loss': reward['loss'],
			# 'learning_tool': learning_tool,
			'hyper_params': learning_tool.hyper_params,
			'model_params': model_params,
			'time_cost': reward['time_cost']
		}

	def update_best_val_score(self, val_score):
		self.evaluation_count += 1
		if val_score > self.cur_best_val_score:
			self.cur_best_val_score = val_score
		if self.evaluation_count % 10 == 1:
			logger.info("====== [{}] Current Best val_{}: {}".format(
				time.time()-self.time_ref, self.evaluation_rule, self.cur_best_val_score))

	def get_learning_curve(self):
		return copy.deepcopy(self.learning_curve)

	def save_learning_curve(self, f_name=None, base_dir=None):
		import pickle
		if base_dir is None:
			base_dir = osp.join(os.path.expanduser('~'), ".das", 'evaluator', 'lcvs')
		if not osp.exists(base_dir):
			os.makedirs(base_dir)
		if f_name is None:
			f_name = "{}.lcv".format(int(time.time()))
		logger.info("Learning Curve Saved in {}".format(osp.join(base_dir, f_name)))
		pickle.dump(self.learning_curve, open(osp.join(base_dir, f_name), 'wb'))

	@staticmethod
	def load_learning_curve(f_name=None, base_dir=None):
		import pickle
		if base_dir is None:
			base_dir = osp.join(os.path.expanduser('~'), ".das", 'evaluator', 'lcvs')
			if not osp.exists(base_dir):
				os.makedirs(base_dir)
		if f_name is None:
			f_name = search_newest_from_dir(base_dir)
		learning_curve = pickle.load(open(osp.join(base_dir, f_name), 'rb'))
		return learning_curve

	@staticmethod
	def extract_time_objective_from_lcv(learning_curve, time_with):
		to_plot_items = list(map(lambda x: (x, learning_curve[x]['{}'.format(time_with)]),
		                         learning_curve.keys()))
		sorted_items = sorted(to_plot_items, key=lambda x: x[0])
		times = list(map(lambda x: x[0], sorted_items))
		objective = list(map(lambda x: x[1], sorted_items))

		if 'val' in time_with:
			max_val_score = -np.inf
			for i in range(len(objective)):
				if i == 0 or objective[i] > max_val_score:
					max_val_score = objective[i]
				objective[i] = max_val_score
		elif 'loss' in time_with:
			min_loss = np.inf
			for i in range(len(objective)):
				if i == 0 or objective[i] < min_loss:
					min_loss = objective[i]
				objective[i] = min_loss
		return times, objective

	@staticmethod
	def plot_single_learning_curve(title="no_title", label="no_label",
	                               learning_curve=None, time_with='val_accuracy_score'):
		if learning_curve is None:
			learning_curve = BaseEvaluator.load_learning_curve()
		if not learning_curve:
			logger.warning("Nothing to plot.")
			return
		import matplotlib.pyplot as plt
		plt.title(title)
		times, objective = BaseEvaluator.extract_time_objective_from_lcv(
			learning_curve, time_with=time_with)
		# plt.yscale('log')
		# plt.ylim([1e-2, 1.0])
		# plt.xscale('log')
		plt.plot(times, objective, label=label)
		plt.xlabel("Time (s)")
		plt.ylabel("{}".format(time_with))
		plt.legend()
		plt.show()

	@staticmethod
	def plot_learning_curves(title="", learning_curves: list=None, time_with='val_accuracy_score', **kwargs):
		assert len(learning_curves) > 0, "number of learning curves should > 0"
		import matplotlib.pyplot as plt
		plt.title(title)
		for lcv_name, learning_curve in learning_curves:
			times, objective = BaseEvaluator.extract_time_objective_from_lcv(
				learning_curve=learning_curve, time_with=time_with)
			logger.debug('label = {}'.format(lcv_name))
			while len(objective) > 0 and objective[0] < 0.0:
				objective = objective[1:]
				times = times[1:]
			plt.plot(times, objective, label=lcv_name)  # color='b' if 'S_1' in lcv_name else 'orange')
		plt.xlabel("Time (s)")
		plt.ylabel("{}".format(time_with))
		if 'yscale' in kwargs:
			plt.yscale(kwargs['yscale'])
		if 'ylim' in kwargs:
			plt.ylim(kwargs['ylim'])
		if 'xscale' in kwargs:
			plt.xscale(kwargs['xscale'])
		# plt.yscale('log')
		# plt.ylim([0.85, 0.875])
		# plt.xscale('log')
		plt.legend()
		plt.show()


class CrossValEvaluator(BaseEvaluator):

	def __init__(self,
	             n_folds=3,
	             evaluation_rule=None,
	             **kwargs):
		super(CrossValEvaluator, self).__init__(validation_strategy='cv', validation_strategy_args=n_folds,
		                                        evaluation_rule=evaluation_rule, **kwargs)

	def evaluate(self, learning_tool: DeepArchiClassifier, X, y, run_time_limit=240.0, random_state=None, **kwargs):
		pass

