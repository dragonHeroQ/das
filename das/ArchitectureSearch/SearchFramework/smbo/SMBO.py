import das
import time
import numpy as np
import logging
from das.util.proba_utils import from_probas_to_performance
from das.ArchitectureSearch.SearchFramework.BaseSearchFramework import BaseSearchFramework

logger = logging.getLogger(das.logger_name)


class SMBO(BaseSearchFramework):

	def __init__(self,
	             optimizer=None,
	             optimizer_class=None,
	             optimizer_params=None,
	             evaluator=None,
	             learning_tool=None,
	             sampling_strategy=None,
	             search_space=None,
	             total_budget=10,
	             budget_type="trial",
	             per_run_timelimit=240.0,
	             evaluation_rule=None,
	             time_ref=None,
	             worst_loss=None,
	             random_state=None,
	             task=None,
	             n_classes=None,
	             **kwargs
	             ):
		if optimizer is not None:
			self.optimizer = optimizer
		else:
			assert learning_tool is not None, "You should set learning_tool for SMBO(BaseSearchFramework)"
			parameter_space = learning_tool.get_parameter_space()
			optimizer_params = {} if optimizer_params is None else optimizer_params
			self.optimizer = optimizer_class(parameter_space=parameter_space, **optimizer_params)
		super(SMBO, self).__init__(optimizer=self.optimizer,
		                           evaluator=evaluator,
		                           learning_tool=learning_tool,
		                           sampling_strategy=sampling_strategy,
		                           search_space=search_space,
		                           total_budget=total_budget,
		                           budget_type=budget_type,
		                           per_run_timelimit=per_run_timelimit,
		                           evaluation_rule=evaluation_rule,
		                           time_ref=time_ref,
		                           worst_loss=worst_loss,
		                           task=task,
		                           random_state=random_state,
		                           **kwargs)
		self.n_classes = n_classes
		self.classes_ = None
		self.records = []
		self.best_config = None
		self.best_num_layers = None

	def handle_y_mapping(self, y):
		if self.task == 'classification':
			self.classes_ = []
			uni_y = np.unique(y)
			for cls in uni_y:
				self.classes_.append(cls)

	def fit(self, X, y, **fit_params):
		"""
		Steps:
			while True:
				1. Judge early stopping criteria
				2. Get next config
				3. Create model from the config
				4. Calculate time left for this trial
				5. Invoke evaluator to evaluate this config
		:param X:
		:param y:
		:param fit_params:
		:return:
		"""
		self.records = []
		self.best_config = None
		self.best_num_layers = None
		if self.sampling_strategy is not None:
			X, y = self.sampling_strategy.sample(X, y, random_state=self.random_state)
		self.handle_y_mapping(y)
		if self.task == 'classification':
			n_classes = self.n_classes or len(np.unique(y))
		else:
			n_classes = 1

		if self.n_classes is None:
			self.n_classes = n_classes
		self.learning_tool.set_classes(n_classes)
		self.time_ref = time.time()
		trial_id = 0
		is_debugging = False
		if 'debug' in fit_params:
			is_debugging = fit_params['debug']
			fit_params.pop('debug')
		while True:
			if ((self.budget_type == 'time' and time.time() - self.time_ref >= self.total_budget - 2)
				  or (self.budget_type == 'trial' and trial_id >= self.total_budget)):  # stopping criteria
				break
			config = self.optimizer.get_next_config(debug=is_debugging)
			learning_tool = self.learning_tool.create_learning_tool(**config)
			time_left = (self.total_budget - 2 - (time.time() - self.time_ref)
			             if self.budget_type == 'time' else self.per_run_timelimit)
			logger.info("Running {}: {}, TimeLimit={}".format(
				trial_id, config, min(time_left, self.per_run_timelimit)))
			reward = self.evaluator.evaluate(learning_tool=learning_tool, X=X, y=y,
			                                 run_time_limit=min(time_left, self.per_run_timelimit),
			                                 random_state=self.random_state, **fit_params)
			logger.info("Config {}: {} --> reward={}".format(trial_id, config, reward))
			self.optimizer.new_result(config=config, reward=reward, other_infos=None, update_model=True)
			self.records.append((config, reward))
			trial_id += 1
		logger.info('SMBO fit TimeCost = {}'.format(time.time() - self.time_ref))
		return self

	def refit(self, X, y, **refit_params):
		pass

	def refit_transform(self, X, y, X_test, **refit_params):
		self.gen_best_record()
		learning_tool = self.learning_tool.create_learning_tool(**self.best_config)
		best_nLayer = None
		run_time_limit = 3600.0
		if 'run_time_limit' in refit_params:
			run_time_limit = refit_params['run_time_limit']
			refit_params.pop('run_time_limit')
		y_pred, y_test_pred = self.evaluator.fit_predict(learning_tool=learning_tool,
		                                                 X=X, y=y, X_test=X_test,
		                                                 run_time_limit=run_time_limit,
		                                                 random_state=self.random_state,
		                                                 best_num_layers=best_nLayer, **refit_params)
		return y_pred, y_test_pred

	def refit_and_score(self, X, y, X_test, y_test, **refit_params):
		y_pred, y_test_pred = self.refit_transform(X, y, X_test, **refit_params)
		trainScore = from_probas_to_performance(y_pred, y, n_classes=self.n_classes,
		                                        task=self.task, evaluation_rule=self.evaluation_rule,
		                                        classes_=self.classes_)
		testScore = from_probas_to_performance(y_test_pred, y_test, n_classes=self.n_classes,
		                                       task=self.task, evaluation_rule=self.evaluation_rule,
		                                       classes_=self.classes_)
		return trainScore, testScore

	def gen_best_record(self):
		sorted_records = sorted(self.records,
		                        key=lambda x: x[1]['val_{}'.format(self.evaluation_rule)],
		                        reverse=True)
		self.best_config = sorted_records[0][0]
		self.best_num_layers = sorted_records[0][1]['best_nLayer']
		logger.info("Best Config: {}".format(self.best_config))
		logger.info("Best Reward: {}".format(sorted_records[0][1]))

	def get_best_n_records(self, n):
		n = min(n, len(self.records))
		sorted_records = sorted(self.records,
		                        key=lambda x: x[1]['val_{}'.format(self.evaluation_rule)],
		                        reverse=True)
		# [(config, reward), (config, reward), ...]
		return sorted_records[:n]

	def get_best_n_learning_tools(self, n):
		best_n_records = self.get_best_n_records(n)
		n_learning_tools = list(map(lambda x: (self.learning_tool.create_learning_tool(**x[0]), x[1]),
		                            best_n_records))
		return n_learning_tools

	def get_learning_curve(self):
		assert self.evaluator is not None, 'evaluator is None, so we have no learning curve yet.'
		return self.evaluator.get_learning_curve()
