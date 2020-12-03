import os
import das
import time
import copy
import logging
import collections
import numpy as np
from das.util.proba_utils import from_probas_to_performance
from das.performance_evaluation import initial_worst_loss
from das.ArchitectureSearch.SearchFramework.evolutionary.EvolutionaryAlgorithm import EvolutionaryAlgorithm

logger = logging.getLogger(das.logger_name)


class AgingEvolution(EvolutionaryAlgorithm):

	def __init__(self, n_classes=None, **kwargs):
		super(AgingEvolution, self).__init__(**kwargs)
		self.population = list()
		self.history = list()
		self.trial_id = 0
		self.classes_ = []
		self.n_classes = n_classes
		self.best_config = None
		self.best_num_layers = None

	def search_from_history(self, archi_config):
		# if have, return reward, else return None
		for cfg, reward in self.history:
			if cfg == archi_config:
				return cfg, reward
		return None, None

	def quality_check(self, archi_config):
		if not hasattr(self, 'failed_pool'):
			self.failed_pool = collections.defaultdict(int)
			self.quality_pool_total = collections.defaultdict(int)
			self.peer_cur = collections.defaultdict(list)
		for key in archi_config:
			if 'algo' in key:
				if self.quality_pool_total[archi_config[key]] >= 2:
					prob = float(self.failed_pool[archi_config[key]]) / self.quality_pool_total[archi_config[key]]
					if np.random.rand() < prob:
						return False
		return True

	def quality_assurance_update(self, archi_config, reward, time_limit, tolerance_exceptions=None):
		if not hasattr(self, 'failed_pool'):
			self.failed_pool = collections.defaultdict(int)
			self.quality_pool_total = collections.defaultdict(int)
			self.peer_cur = collections.defaultdict(list)
		if tolerance_exceptions is None:
			tolerance_exceptions = []
		success = 1
		if reward is None:
			success = 0
		elif 'time_cost' in reward and reward['time_cost'] >= time_limit:
			success = 0
		elif 'loss' in reward and reward['loss'] < initial_worst_loss(self.evaluation_rule):
			success = 1
		elif ('exception' in reward
		      and reward['exception'] is not None
		      and reward['exception'] not in tolerance_exceptions):
			success = 0

		if success == 0:
			have_monster = False
			for key in archi_config:
				if 'algo' in key:
					if archi_config[key] in ['SVC', 'SVR', 'GPClassifier', 'GPRegressor']:
						have_monster = True
			for key in archi_config:
				if 'algo' in key:
					if have_monster and archi_config[key] in ['SVC', 'SVR', 'GPClassifier', 'GPRegressor']:
						self.failed_pool[archi_config[key]] += 1
					elif have_monster and archi_config[key] not in ['SVC', 'SVR', 'GPClassifier', 'GPRegressor']:
						self.failed_pool[archi_config[key]] += 0
					elif not have_monster:
						self.failed_pool[archi_config[key]] += 1
					self.quality_pool_total[archi_config[key]] += 1
		else:
			for key in archi_config:
				if 'algo' in key:
					self.failed_pool[archi_config[key]] += 0
					self.quality_pool_total[archi_config[key]] += 1

		print("=================FAILED TABLE====================")
		for key in self.failed_pool:
			print("{}: {}/{}".format(key, self.failed_pool[key], self.quality_pool_total[key]))
		print("+++++++++++++++++FAILED TABLE++++++++++++++++++++")

	def init_population(self, X, y, **fit_params):
		"""
		Init_population 阶段不允许重复！

		:param X:
		:param y:
		:param fit_params:
		:return:
		"""
		while len(self.population) < self.P:
			if ((self.budget_type == 'time' and time.time() - self.time_ref >= self.total_budget - 2)
					or (self.budget_type == 'trial' and self.trial_id >= self.total_budget)):  # stopping criteria
				break
			while True:
				archi_config = self.optimizer.get_next_config()
				print("ArchiConfig = {}".format(archi_config))
				if self.quality_check(archi_config):
					break
				else:
					logger.info("[Quality Assurance] REJECT config {}".format(archi_config))
			learning_tool = self.learning_tool.create_learning_tool(
				cross_validator=self.cross_validator, **archi_config)
			archi_config['params'] = learning_tool.learning_estimator.get_params()
			time_left = (self.total_budget - 2 - (time.time() - self.time_ref)
			             if self.budget_type == 'time' else self.per_run_timelimit)
			run_time_limit = min(time_left, self.per_run_timelimit)
			os.system("df -h | grep '/dev/shm'")
			logger.info("Running {}: {}={}, TimeLimit={}".format(self.trial_id,
			                                                     self.encode_archi(archi_config,
			                                                                       learning_tool=learning_tool),
			                                                     archi_config,
			                                                     run_time_limit))

			cfg, history_answer = self.search_from_history(archi_config)
			if history_answer is not None:
				reward = history_answer
				logger.info("HISTORY DISCOVERED!!!")
				continue
			else:
				reward = self.evaluator.evaluate(
					learning_tool=learning_tool, X=X, y=y,
					run_time_limit=run_time_limit, random_state=self.random_state, **fit_params)
			# reward = {'val_{}'.format(self.evaluation_rule): np.random.rand()}
			logger.info("Config {}=[{}]: {} --> reward={}".format(
				self.trial_id, self.encode_archi(archi_config, learning_tool=learning_tool), archi_config, reward))
			self.quality_assurance_update(archi_config=archi_config, reward=reward,
			                              time_limit=run_time_limit, tolerance_exceptions=[])
			# ===== debug ray ======
			copy_archi = copy.deepcopy(archi_config)
			if 'params' in copy_archi:
				copy_archi.pop('params')
			print("{} TIMECOST {}, REWARD {}".format(
				copy_archi, reward['time_cost'], reward['val_{}'.format(self.evaluation_rule)]))
			# ===== debug ray ======
			self.population.append((self.encode_archi(archi_config, learning_tool=learning_tool), reward))
			self.history.append((archi_config, reward))
			self.optimizer.new_result(config=archi_config, reward=reward, other_infos=None, update_model=True)
			self.trial_id += 1
		logger.info("Population initialized!")

	def eliminate_member(self):
		self.population.pop(0)

	def encode_archi(self, archi_config, **kwargs):
		return self.learning_tool.encode_archi(archi_config, **kwargs)

	def decode_archi(self, encoded_archi):
		return self.learning_tool.decode_archi(encoded_archi)

	def mutate(self, encoded_archi, learning_tool, **kwargs):
		if np.random.rand() < self.identity_proba:  # identity mutation
			return encoded_archi
		candidate_mutation_ops = self.learning_tool.mutation_ops()
		num_ops = len(candidate_mutation_ops)
		choice_ops = candidate_mutation_ops[np.random.randint(0, num_ops)]
		mutated_string = choice_ops(
			encoded_archi, is_classification=(self.task == 'classification'), learning_tool=learning_tool)
		return mutated_string

	def handle_y_mapping(self, y):
		self.classes_ = []
		if self.task == 'classification':
			uni_y = np.unique(y)
			for cls in uni_y:
				self.classes_.append(cls)

	def fit(self, X, y, **fit_params):
		if 'debug' in fit_params:
			if fit_params['debug'] is True:
				logger.setLevel('DEBUG')
			fit_params.pop('debug')
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
		self.best_config = None
		self.best_num_layers = None

		# initialize cross validation cache
		# deprecated
		# if self.cross_validator is not None:
		# 	raise Exception("Now we do not support cross_validator")
		# 	self.cross_validator.init_cache(X, y, X_follow=None, cv=self.evaluator.n_folds,
		# 	                                task=self.task, random_state=self.random_state,
		# 	                                redis_address=self.evaluator.redis_address)
		self.init_population(X, y, **fit_params)

		while True:
			if ((self.budget_type == 'time' and time.time() - self.time_ref >= self.total_budget - 2)
					or (self.budget_type == 'trial' and self.trial_id >= self.total_budget)):  # stopping criteria
				break
			sample = []
			while len(sample) < self.S:
				ind = np.random.randint(0, len(self.population))
				sample.append(self.population[ind])
			parent_id = _get_parent_with_highest_score(sample, self.evaluation_rule)
			# print("self.population[parent_id][0]", self.population[parent_id][0])
			tmp_learning_tool = self.learning_tool.create_learning_tool(
				cross_validator=self.cross_validator,
				**self.decode_archi(self.population[parent_id][0]))
			print("========PARENT and SCORE=========")
			print(self.decode_archi(self.population[parent_id][0]))
			print("=================================")
			cnt = 0
			while True:
				cnt += 1
				encoded_child_archi = self.mutate(self.population[parent_id][0], learning_tool=tmp_learning_tool)
				decoded_child_archi_config = self.decode_archi(encoded_child_archi)
				if self.quality_check(decoded_child_archi_config):
					break
				else:
					logger.info("[Quality Assurance] REJECT config {}".format(decoded_child_archi_config))
				if cnt >= 20:
					parent_id = np.random.randint(0, len(self.population))
					logger.info("[Quality Assurance] No appropriate mutation, change parent to {}".format(parent_id))
				if cnt >= 200:  # max tolerance
					break
			logger.info('Cycle {}, parent {} => child {}'.format(self.trial_id,
			                                                     self.population[parent_id][0],
			                                                     encoded_child_archi))
			archi_config = self.decode_archi(encoded_archi=encoded_child_archi)
			learning_tool = self.learning_tool.create_learning_tool(
				cross_validator=self.cross_validator, **archi_config)
			time_left = (self.total_budget - 2 - (time.time() - self.time_ref)
			             if self.budget_type == 'time' else self.per_run_timelimit)
			run_time_limit = min(time_left, self.per_run_timelimit)
			os.system("df -h | grep '/dev/shm'")
			logger.info("Running {}: {}={}, TimeLimit={}".format(
				self.trial_id, encoded_child_archi, archi_config, run_time_limit))
			archi_config['params'] = learning_tool.learning_estimator.get_params()
			cfg, history_answer = self.search_from_history(archi_config)
			if history_answer is not None:
				reward = history_answer
				logger.info("HISTORY DISCOVERED!!!")
			else:
				reward = self.evaluator.evaluate(learning_tool=learning_tool,
				                                 X=X, y=y, run_time_limit=run_time_limit,
				                                 random_state=self.random_state, **fit_params)
			# reward = {'val_{}'.format(self.evaluation_rule): np.random.rand()}
			logger.info("Config {}=[{}]: {} --> reward={}".format(
				self.trial_id, encoded_child_archi, archi_config, reward))
			self.quality_assurance_update(archi_config=archi_config, reward=reward,
			                              time_limit=run_time_limit, tolerance_exceptions=[])
			# ===== debug ray ======
			copy_archi = copy.deepcopy(archi_config)
			if 'params' in copy_archi:
				copy_archi.pop('params')
			print("{} TIMECOST {}, REWARD {}".format(
				copy_archi, reward['time_cost'], reward['val_{}'.format(self.evaluation_rule)]))
			# ===== debug ray ======
			self.population.append((self.encode_archi(archi_config, learning_tool=learning_tool), reward))
			self.history.append((archi_config, reward))
			self.eliminate_member()
			self.optimizer.new_result(config=archi_config, reward=reward, other_infos=None, update_model=True)
			self.trial_id += 1
		logger.info('AgingEvolution fit TimeCost = {}'.format(time.time() - self.time_ref))
		return self

	def gen_best_record(self):
		sorted_records = sorted(self.history,
		                        key=lambda x: x[1]['val_{}'.format(self.evaluation_rule)],
		                        reverse=True)
		self.best_config = sorted_records[0][0]
		self.best_num_layers = sorted_records[0][1]['best_nLayer']
		logger.info("Best Config: {}".format(self.best_config))
		logger.info("Best Reward: {}".format(sorted_records[0][1]))

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


def _get_parent_with_highest_score(sample, evaluation_rule):
	max_score = sample[0][1]['val_{}'.format(evaluation_rule)]
	best_ind = 0
	for i in range(1, len(sample)):
		if sample[i][1]['val_{}'.format(evaluation_rule)] > max_score:
			max_score = sample[i][1]['val_{}'.format(evaluation_rule)]
			best_ind = i
	return best_ind


if __name__ == '__main__':
	from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
	from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator
	# from das.ArchitectureSearch.LearningTool.FullDeepArchiLearningTool import FullDeepArchiLearningTool
	from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool

	n_classes = 2
	evaluation_rule = 'accuracy_score'
	learning_tool = DeepArchiLearningTool(n_classes=n_classes, evaluation_rule=evaluation_rule)
	evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule=evaluation_rule)
	ae = AgingEvolution(n_classes=n_classes, learning_tool=learning_tool, evaluator=evaluator,
	                    budget_type='trial', total_budget=5,
	                    optimizer_class=RandomSearchOptimizer, evaluation_rule=evaluation_rule)
	ae.fit(X=1, y=1)
	ae.refit_and_score(1, 1, 1, 1)
