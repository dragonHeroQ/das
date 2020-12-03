import das
import ray
import time
import logging
import numpy as np
from das.util.proba_utils import from_probas_to_performance
from das.ArchitectureSearch.SearchFramework.evolutionary.EvolutionaryAlgorithm import EvolutionaryAlgorithm

logger = logging.getLogger(das.logger_name)
np.random.seed(0)


class DisAgingEvolution(EvolutionaryAlgorithm):

	def __init__(self, n_classes=None, **kwargs):
		super(DisAgingEvolution, self).__init__(**kwargs)
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

	def init_population(self, X, y, **fit_params):
		learning_tools = []
		archi_configs = []
		time_start = time.time()
		while len(learning_tools) < self.P:
			if ((self.budget_type == 'time' and time.time() - self.time_ref >= self.total_budget - 2)
				  or (self.budget_type == 'trial' and self.trial_id >= self.total_budget)):  # stopping criteria
				break
			archi_config = self.optimizer.get_next_config()
			learning_tool = self.learning_tool.create_learning_tool(**archi_config)
			archi_config['params'] = learning_tool.learning_estimator.get_params()
			# time_left = (self.total_budget - 2 - (time.time() - self.time_ref)
			#              if self.budget_type == 'time' else self.per_run_timelimit)
			# logger.info("Running {}: {}={}, TimeLimit={}".format(self.trial_id,
			#                                                      self.encode_archi(archi_config,
			#                                                                        learning_tool=learning_tool),
			#                                                      archi_config,
			#                                                      min(time_left, self.per_run_timelimit)))

			cfg, history_answer = self.search_from_history(archi_config)
			if history_answer is not None:
				reward = history_answer
				logger.info("HISTORY DISCOVERED!!!")
				learning_tools.append(reward)
			else:
				learning_tools.append(learning_tool)
				# reward = self.evaluator.evaluate(
				# 	learning_tool=learning_tool, X=X, y=y,
				# 	run_time_limit=self.per_run_timelimit, **fit_params)
				# reward = {'val_{}'.format(self.evaluation_rule): np.random.rand()}
			archi_configs.append(archi_config)
		total_left_time = self.total_budget - 2 - (time.time() - self.time_ref)
		time_left = (total_left_time if self.budget_type == 'time' else self.per_run_timelimit)
		eval_results = self.evaluator.evaluate(learning_tools=learning_tools,
		                                       X=X, y=y, run_time_limit=time_left,
		                                       random_state=self.random_state,
		                                       total_time=total_left_time, **fit_params)
		for i, reward in enumerate(eval_results):
			if reward is None:
				continue
			logger.info("Config {}=[{}]: {} --> reward={}".format(
				i, self.encode_archi(archi_configs[i], learning_tool=learning_tools[i]),
				archi_configs[i], reward))
			self.population.append((self.encode_archi(archi_configs[i], learning_tool=learning_tools[i]), reward))
			self.history.append((archi_configs[i], reward))
			self.optimizer.new_result(config=archi_configs[i], reward=reward, other_infos=None, update_model=True)
			# self.trial_id += 1
		logger.info("Population initialized! Time Cost: {}".format(time.time()-time_start))

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
			tmp_learning_tool = self.learning_tool.create_learning_tool(**self.decode_archi(self.population[parent_id][0]))
			encoded_child_archi = self.mutate(self.population[parent_id][0], learning_tool=tmp_learning_tool)
			logger.info('Cycle {}, parent {} => child {}'.format(self.trial_id,
			                                                     self.population[parent_id][0],
			                                                     encoded_child_archi))
			archi_config = self.decode_archi(encoded_archi=encoded_child_archi)
			learning_tool = self.learning_tool.create_learning_tool(**archi_config)
			time_left = (self.total_budget - 2 - (time.time() - self.time_ref)
			             if self.budget_type == 'time' else self.per_run_timelimit)
			logger.info("Running {}: {}={}, TimeLimit={}".format(
				self.trial_id, encoded_child_archi, archi_config, min(time_left, self.per_run_timelimit)))
			archi_config['params'] = learning_tool.learning_estimator.get_params()
			cfg, history_answer = self.search_from_history(archi_config)
			if history_answer is not None:
				reward = history_answer
				logger.info("HISTORY DISCOVERED!!!")
			else:
				reward = self.evaluator.evaluate(learning_tool=learning_tool,
				                                 X=X, y=y, run_time_limit=time_left,
				                                 **fit_params)
				if isinstance(reward, ray.ObjectID):
					reward = ray.get(reward)
				# reward = {'val_{}'.format(self.evaluation_rule): np.random.rand()}
			logger.info("Config {}=[{}]: {} --> reward={}".format(
				self.trial_id, encoded_child_archi, archi_config, reward))
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
	from das.ArchitectureSearch.LearningTool.FullDeepArchiLearningTool import FullDeepArchiLearningTool
	n_classes = 2
	evaluation_rule = 'accuracy_score'
	learning_tool = FullDeepArchiLearningTool(n_classes=n_classes, evaluation_rule=evaluation_rule)
	evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule=evaluation_rule)
	ae = AgingEvolution(n_classes=n_classes, learning_tool=learning_tool,
	                    budget_type='time', total_budget=3,
	                    optimizer_class=RandomSearchOptimizer, evaluation_rule=evaluation_rule)
	ae.fit(X=1, y=1)
	ae.refit_and_score(1, 1, 1, 1)

