import das
import copy
import time
import logging
import numpy as np
from das.ArchitectureSearch.SearchFramework.hyperband.Bandits import Bandit
from das.util.proba_utils import from_probas_to_performance
from das.performance_evaluation import eval_performance, initial_worst_loss, judge_rule
from das.util.common_utils import min_max_scale, soft_max, flip_coin
from das.ArchitectureSearch.SearchFramework.hyperband.HyperBandRelated import StageTimeController

logger = logging.getLogger(das.logger_name)


class HyperBandIteration(object):

	def __init__(self,
	             iteration_budget=None,
	             budget_type='time',
	             all_bandits: dict=None,
	             evaluator=None,
	             sampling_strategy=None,
	             stage_time_controller: StageTimeController=None,
	             max_num_stages=None,
	             time_ref=None,
	             evaluation_rule=None,
	             per_run_timelimit=240.0,
	             n_classes=None,
	             task=None,
	             worst_loss=None,
	             random_state=None,
	             ):
		self.iteration_budget = iteration_budget
		self.budget_type = budget_type
		assert self.budget_type in ['trial', 'time'], \
			"Unsupported budget type: {}".format(self.budget_type)
		self.all_bandits = all_bandits
		# if bandit: BaseLearningTool is passes in, transform it to Bandit automatically
		for key in self.all_bandits.keys():
			if not isinstance(self.all_bandits[key], Bandit):
				self.all_bandits[key] = Bandit(learning_tool=self.all_bandits[key],
				                               algorithm_model=self.all_bandits[key].learning_estimator)
		self.num_bandits = len(self.all_bandits)
		self.evaluator = evaluator
		self.sampling_strategy = sampling_strategy
		self.stage_time_controller = stage_time_controller
		self.max_num_stages = max_num_stages
		self.time_ref = time_ref or time.time()
		self.evaluation_rule = evaluation_rule
		self.per_run_timelimit = per_run_timelimit
		self.n_classes = n_classes
		self.task = task or judge_rule(self.evaluation_rule)
		self.worst_loss = worst_loss or initial_worst_loss(rule=self.evaluation_rule)
		self.random_state = random_state
		self.classes_ = []
		self.budget_for_stages = [self.iteration_budget / self.max_num_stages
		                          for _ in range(self.max_num_stages)]
		self.current_live_list = set()
		if self.all_bandits is not None:
			self.current_live_list.update(self.all_bandits.keys())
		self.reject_list = set()

		# running attrs
		self.key_of_best_bandit = None
		self.best_bandit_hyperparam = None
		self.best_bandit = None
		self.best_score = None
		self.num_failed_trials = 0
		self.total_trials = 0
		self.fitted = False

	def allocate_budget_for_bandits(self, stage_budget):
		raise NotImplementedError

	def _get_bandit_score(self):
		self.all_bandit_ucb_score = {}
		for k in self.current_live_list:
			self.all_bandit_ucb_score[k] = self.all_bandits[k].get_best_score() or 0
		return self.all_bandit_ucb_score

	def do_halving(self):
		raise NotImplementedError

	def update_failed_trials(self, rewards):
		for reward in rewards:
			if 'exception' in reward and reward['exception'] is not None:
				self.num_failed_trials += 1
			self.total_trials += 1

	def handle_y_mapping(self, y):
		self.classes_ = []
		if self.task == 'classification':
			uni_y = np.unique(y)
			for cls in uni_y:
				self.classes_.append(cls)

	def fit(self, X, y, **fit_params):
		if self.fitted:
			raise Exception("Now we do not support duplicated fit!")
		if self.sampling_strategy is not None:
			X, y = self.sampling_strategy.sample(X, y, random_state=self.random_state)
		self.handle_y_mapping(y)
		if self.task == 'classification':
			n_classes = self.n_classes or len(np.unique(y))
		else:
			n_classes = 1
		if self.n_classes is None:
			self.n_classes = n_classes

		for i in range(0, self.max_num_stages):
			# for every stage
			stage_budget = self.budget_for_stages[i]
			assert stage_budget > 0, "stage_budget = {}, not enough!".format(stage_budget)
			budget_for_every_bandit = self.allocate_budget_for_bandits(stage_budget)
			print("resource for every bandit", budget_for_every_bandit)

			for k in self.current_live_list:
				if budget_for_every_bandit[k] <= 0.0:  # no budget, no need to start
					continue
				logger.info("Running {} ...".format(
					self.all_bandits[k].algorithm_model.get_model_name(concise=True)))
				(model_parameters, rewards) = self.all_bandits[k].compute_deep_archi(
					evaluator=self.evaluator, X=X, y=y, budget=budget_for_every_bandit[k],
					budget_type=self.budget_type, per_run_timelimit=self.per_run_timelimit,
					random_state=self.random_state, **fit_params
				)
				# (rewards, model_parameters) = self.all_bandits[k].compute(
				# 	X=X, y=y, X_val=None, y_val=None, budget=budget_for_every_bandit[k], budget_type=self.budget_type,
				# 	validation_strategy='cv', validation_strategy_args=3, per_run_timebudget=self.per_run_timelimit,
				# 	evaluation_rule=self.evaluation_rule, time_ref=self.time_ref, worst_loss=self.worst_loss,
				# 	n_classes=self.n_classes, random_state=self.random_state)
				self.update_failed_trials(rewards)
				self.all_bandits[k].add_records(model_parameters, rewards)

			before_reject_current = len(self.current_live_list)
			self.do_halving()
			logger.info("Round {}, from {} bandits to {} bandits".format(
				i, before_reject_current, len(self.current_live_list)))
		self.fitted = True
		print("fit time cost: {}".format(time.time()-self.time_ref))
		return self

	def get_best_bandit(self):
		return self.best_bandit

	def get_key_of_best_bandit(self):
		return self.key_of_best_bandit

	def get_all_bandits(self):
		return copy.deepcopy(self.all_bandits)

	def configure_bandits(self, all_bandits):
		if not isinstance(all_bandits, dict):
			raise Exception("The parameter all_bandits should be a dict")
		self.all_bandits = all_bandits

	def propose_best_bandit_and_hyperparam(self):
		best_val_loss = None
		best_ind = -1
		best_para = None
		for i in self.all_bandits.keys():
			if best_val_loss is None or (self.all_bandits[i].get_best_score() is not None
			                             and best_val_loss > self.all_bandits[i].get_best_score()):
				best_val_loss = self.all_bandits[i].get_best_score()
				best_ind = i
				best_para = self.all_bandits[i].get_best_model_parameters()

		self.best_score = best_val_loss
		self.key_of_best_bandit = best_ind
		self.best_bandit_hyperparam = best_para
		self.best_bandit = copy.deepcopy(self.all_bandits[best_ind])
		print("best ind: {}".format(best_ind))
		print("best val loss: ", best_val_loss)
		print("best bandit: {}".format(self.all_bandits[best_ind].algorithm_model.get_model_name()))
		print("best parameter: {}".format(best_para))

	def refit(self, X, y):
		self.propose_best_bandit_and_hyperparam()
		self.best_bandit.algorithm_model.set_params(**self.best_bandit_hyperparam)
		self.best_bandit.algorithm_model.fit(X, y)

	def refit_transform(self, X, y, X_test, **refit_params):
		self.propose_best_bandit_and_hyperparam()
		self.best_bandit.algorithm_model.set_params(**self.best_bandit_hyperparam)
		learning_tool = self.best_bandit.learning_tool
		# TODO: get best_num_layers if there is no sampling performed
		run_time_limit = 3600.0
		if 'run_time_limit' in refit_params:
			run_time_limit = refit_params['run_time_limit']
			refit_params.pop('run_time_limit')
		y_pred, y_test_pred = self.evaluator.fit_predict(learning_tool, X, y, X_test, run_time_limit=run_time_limit,
		                                                 random_state=self.random_state, **refit_params)
		# y_pred, y_test_pred = estimator.fit_predict(X, y, X_test, random_state=self.random_state,
		#                                             best_num_layers=None, **refit_params)
		return y_pred, y_test_pred

	def refit_and_score(self, X, y, X_test, y_test, **refit_params):
		y_pred, y_test_pred = self.refit_transform(X, y, X_test, **refit_params)
		trainScore = from_probas_to_performance(y_pred, y, n_classes=self.n_classes, task=self.task,
		                                        evaluation_rule=self.evaluation_rule, classes_=self.classes_)
		testScore = from_probas_to_performance(y_test_pred, y_test, n_classes=self.n_classes, task=self.task,
		                                       evaluation_rule=self.evaluation_rule, classes_=self.classes_)
		return trainScore, testScore

	def predict(self, X):
		assert self.best_bandit is not None, "You should firstly refit, and then predict"
		y_prediction = self.best_bandit.algorithm_model.predict(X)
		return y_prediction

	def score(self, X, y):
		assert self.best_bandit is not None, "You should firstly refit, and then predict"
		y_prediction = self.best_bandit.algorithm_model.predict(X)
		print("Final Test Score: {}".format(eval_performance(self.evaluation_rule,
		                                                     y_true=y, y_score=y_prediction)))

	def print_summary(self):
		print("Total Actions: {}".format(
			sorted([x.get_num_of_actions() for x in self.all_bandits.values()], reverse=True)))


class UCBIteration(HyperBandIteration):

	def allocate_budget_for_bandits(self, stage_budget):
		tmp_ucb = {}
		for i in self.current_live_list:
			tmp_ucb[i] = self.all_bandits[i].get_ucb_score() or 1
		tmp_ucb = min_max_scale(tmp_ucb)  # scale to assign higher ucb score to smaller loss
		pi = soft_max(tmp_ucb)
		for i in pi.keys():
			pi[i] = pi[i] * stage_budget
			if self.budget_type == 'trial':
				pi[i] = int(pi[i])
		return pi

	def _get_ucb_score(self):
		self.all_bandit_ucb_score = {}
		for k in self.current_live_list:
			self.all_bandit_ucb_score[k] = self.all_bandits[k].get_ucb_score() or 0
		return self.all_bandit_ucb_score

	def do_halving(self):
		# reject one bandit at least
		tmp_ucb_score = self._get_ucb_score()

		tmp_success_rate = {}
		for i in self.current_live_list:
			tmp_success_rate[i] = tmp_ucb_score[i]

		tmp_success_rate = min_max_scale(tmp_success_rate)
		print("REJECTING: tmp_success_Rate = {}".format(tmp_success_rate))
		self.current_live_list.clear()
		newly_rejected = 0

		while True:
			for i in tmp_success_rate.keys():
				# reject based on probability
				should_stop = not flip_coin(tmp_success_rate[i])
				if should_stop:
					# reject
					self.reject_list.add(i)
					newly_rejected += 1
				else:
					# go on
					self.current_live_list.add(i)
			# should reject at least one candidate, should leave at least one survivor
			if len(self.current_live_list) > 0 and len(self.reject_list) > 0:
				break

		print("Newly REJECTED {}/{}, LEAVE {}".format(
			newly_rejected, len(self.reject_list), len(self.current_live_list)))


class SuccessiveHalvingIteration(HyperBandIteration):

	def __init__(self, eta=3, **kwargs):
		super(SuccessiveHalvingIteration, self).__init__(**kwargs)
		self.eta = eta

	def allocate_budget_for_bandits(self, stage_budget):
		num_live_bandits = len(self.current_live_list)
		assert num_live_bandits > 0, "No bandit lived, it seems finished?"
		pi = 1.0 / num_live_bandits
		budgets = {}
		for i in self.current_live_list:
			budgets[i] = pi * stage_budget
			if self.budget_type == 'trial':
				budgets[i] = int(budgets[i])
		return budgets

	def do_halving(self):
		# reject one bandit at least
		tmp_bandit_score = self._get_bandit_score()

		tmp_success_rate = {}
		for i in self.current_live_list:
			tmp_success_rate[i] = tmp_bandit_score[i]

		tmp_success_rate = min_max_scale(tmp_success_rate)
		print("REJECTING: tmp_success_Rate = {}".format(tmp_success_rate))
		current_live_bandits = len(self.current_live_list)
		next_stage_live = current_live_bandits // self.eta
		if next_stage_live == 0:
			next_stage_live += 1
		self.current_live_list.clear()
		newly_rejected = 0

		sorted_bandits = sorted(tmp_success_rate.items(), key=lambda x: x[1])
		for b_id, b_score in sorted_bandits[:next_stage_live]:
			self.current_live_list.add(b_id)

		for b_id, b_score in sorted_bandits[next_stage_live:]:
			self.reject_list.add(b_id)
			newly_rejected += 1

		print("Newly REJECTED {}/{}, LEAVE {}".format(
			newly_rejected, len(self.reject_list), len(self.current_live_list)))


if __name__ == '__main__':
	from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
	from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator
	from das.ArchitectureSearch.Optimizer.BayesianOptimizer import BayesianOptimizer
	learning_tool = DeepArchiLearningTool(n_block=2, n_classes=10, evaluation_rule='accuracy_score')
	evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule='accuracy_score')
	optimizer = BayesianOptimizer(parameter_space=learning_tool.get_parameter_space())

	num_bandits = 10
	all_bandits = dict([(i,
	                     learning_tool.create_learning_tool(**optimizer.get_next_config()))
	                    for i in range(10)])

	# debug at 01/22 15:37
	# all_bandits = {0: learning_tool.create_learning_tool(**{'b1_algo': 'RandomForestClassifier', 'b1_num': 4,
	#                                                         'b2_algo': 'GPClassifier', 'b2_num': 2})}
	# print(all_bandits)
	# for bandit in all_bandits.values():
	# 	print(bandit.get_configuration_space().get_space_names())

	ucb_iter = UCBIteration(iteration_budget=1200, budget_type='time',
	                        all_bandits=all_bandits, evaluator=evaluator,
	                        sampling_strategy=None, stage_time_controller=None, max_num_stages=3,
	                        evaluation_rule='accuracy_score',
	                        per_run_timelimit=240.0, random_state=0)
	# logger.setLevel('DEBUG')
	from benchmarks.data.digits.load_digits import load_digits

	x_train, x_test, y_train, y_test = load_digits()
	print(x_train.shape)
	ucb_iter.fit(x_train, y_train, debug=True)
	# learning_curve = evaluator.load_learning_curve()
	# print(learning_curve)
	evaluator.plot_single_learning_curve()
	# evaluator.save_learning_curve()
	train_score, test_score = ucb_iter.refit_and_score(x_train, y_train, x_test, y_test)
	print("Final Train Score={}, Test Score={}".format(train_score, test_score))

