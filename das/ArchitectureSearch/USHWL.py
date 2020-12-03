import sys
import time
import copy
import psutil
import pickle
import numpy as np
import multiprocessing
from das.compmodel import CompositeModel
from das.BaseAlgorithm.algorithm_space import (get_algorithm_by_key,
                                               get_all_regression_algorithm_keys)
from das.crossvalidate import cross_validate_score
from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
from das.performance_evaluation import eval_performance, loss_to_score, score_to_loss


class BanditWL(object):

	def __init__(self, algorithm_model=None, c=2, ucb_mean=0, ucb_var=1e-5):

		self.algorithm_model = algorithm_model

		# records
		self.records = []
		self.model_parameters = []

		# mean and variation
		self.u = ucb_mean
		self.v = ucb_var

		# constant to control gaussian ucb
		self.c = c
		self.num_of_actions = 0
		self.reward = None

	def get_mean(self):
		if len(self.records) > 0:
			self.u = np.mean(self.records)
		return self.u

	def get_variation(self):
		if len(self.records) > 0:
			self.v = np.std(self.records)
		return self.v

	def get_num_of_actions(self):
		return self.num_of_actions

	def _gaussian_ucb_score(self):
		# gaussian ucb
		import math
		tmp_score = self.get_mean() + self.c * self.get_variation() / math.log(self.num_of_actions + 2, 2)
		return tmp_score

	def _best_record(self):
		if len(self.records) == 0:
			return None
		# get the record with the minimum loss
		return min(self.records)

	def get_ucb_score(self):
		# return self._best_record()
		return self._gaussian_ucb_score()

	def add_records(self, in_records, model_parameters=None):
		self.records.extend(in_records)
		self.num_of_actions += len(in_records)
		if model_parameters is not None:
			self.model_parameters.extend(model_parameters)

	@staticmethod
	def cv_score(model, X, y, cv_fold, evaluation_rule, return_dict, random_state=None):
		return_dict["val_score"] = None
		val_score, _ = cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule,
		                                    random_state=random_state)
		return_dict["model"] = model
		return_dict["val_score"] = val_score

	@staticmethod
	def holdout_score(model, X, y, X_val, y_val, evaluation_rule, return_dict, random_state=None):
		return_dict["val_score"] = None
		model.fit(X, y)
		y_hat = model.predict(X_val)
		val_score = eval_performance(rule=evaluation_rule, y_true=y_val, y_score=y_hat,
		                             random_state=random_state)
		return_dict["model"] = model
		return_dict["val_score"] = val_score

	def compute(self, X, y, X_val=None, y_val=None, budget=3, budget_type="trial", hyper_param_getter=None,
	            validation_strategy='cv', validation_strategy_args=3, per_run_timebudget=360.0,
	            evaluation_rule=None, time_ref=None, worst_loss=None, random_state=None):

		model = self.algorithm_model
		assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"

		res = []
		model_parameters = []
		tmp_start_time = time.time()
		num_trials = 0
		if time_ref is None:
			time_ref = time.time()
		learning_curve = {}

		while True:
			failed_flag = False
			tps = self.algorithm_model.get_configuration_space()
			rs = RandomSearchOptimizer(tps)
			candidate_config = rs.get_random_config()
			print("Starting Config: {}".format(candidate_config))
			model_parameters.append(candidate_config)
			self.algorithm_model.set_params(**candidate_config)

			mgr = multiprocessing.Manager()
			return_dict = mgr.dict()

			try:
				if X_val is not None:  # holdout
					assert validation_strategy == 'holdout', 'Validation Strategy is not holdout, why X_val provided?'
					p = multiprocessing.Process(target=self.holdout_score, args=(model, X, y, X_val, y_val,
					                                                             evaluation_rule, return_dict,
					                                                             random_state))
				else:
					assert validation_strategy == 'cv', 'Validation Strategy should be cv if X_val is not provided'
					cv_fold = validation_strategy_args
					assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
					p = multiprocessing.Process(target=self.cv_score, args=(model, X, y, cv_fold, evaluation_rule,
					                                                        return_dict, random_state))
				p.start()
				if budget_type == 'time':
					per_runtime_limit = budget - (time.time() - tmp_start_time)
				else:
					per_runtime_limit = per_run_timebudget
				print("Time Limit: {}".format(per_runtime_limit))
				p.join(per_runtime_limit)

				if p.is_alive():
					p.terminate()
					kill_tree(p.pid)

				if return_dict["val_score"] is not None:
					val_score = return_dict["val_score"]
				else:
					failed_flag = True   # exception occurred
					val_score = loss_to_score(evaluation_rule, worst_loss)

				print("val_score {}".format(val_score))

				loss = score_to_loss(evaluation_rule, val_score)
				self.reward = {'loss': loss,
				               'info': {'val_{}'.format(evaluation_rule): val_score}}
				if worst_loss < loss:   # local worst loss update
					worst_loss = loss + 1
			except Exception as e:
				print(e)
				failed_flag = True
				self.reward = {'loss': worst_loss,
				               'info': {'val_{}'.format(evaluation_rule): loss_to_score(evaluation_rule, worst_loss)}}
			finally:
				learning_curve[time.time()-time_ref] = self.reward['loss']
				res.append(self.reward['loss'])
				if failed_flag:
					print("Exception Occurred!")
			num_trials += 1

			if ((budget_type == 'time' and time.time() - tmp_start_time >= budget-2)
				  or (budget_type == 'trial' and num_trials >= budget)):   # stopping criteria
				break

		return res, model_parameters, learning_curve

	def get_best_record(self):
		return self._best_record()

	def get_best_model_parameters(self):
		if len(self.model_parameters) < 1 or len(self.records) < 1:
			return None
		ind = int(np.argmin(np.array(self.records)))
		return self.model_parameters[ind]


class USHWL(object):

	def __init__(self, total_budget, all_bandits, budget_type="trial", max_number_of_round=3,
	             per_run_timelimit=240.0, evaluation_rule='mean_squared_error', time_ref=None, worst_loss=None):

		self.budget_type = budget_type
		self.total_budget = total_budget
		self.per_run_timelimit = per_run_timelimit
		self.all_bandits = all_bandits

		self.keys_of_all_bandits = all_bandits.keys()
		self.all_bandit_ucb_score = {}
		self.reject_list = set()
		self.current_list = set(self.all_bandits.keys())

		self.evaluation_rule = evaluation_rule
		self.time_ref = time_ref or time.time()
		self.worst_loss = worst_loss or get_worst_loss_from_evaluation_rule(evaluation_rule=evaluation_rule)
		self.learning_curve = {}
		self.max_number_of_round = max_number_of_round
		self.key_of_best_bandit = None
		self.best_bandit_hyperparam = None
		self.best_bandit = None
		self.best_score = None

	def get_learning_curve(self):
		return self.learning_curve

	def get_all_bandits(self):
		return self.all_bandits

	def configure_bandits(self, all_bandits):
		if not isinstance(all_bandits, dict):
			raise Exception("The parameter all_bandits should be a dict")
		self.all_bandits = all_bandits

	@staticmethod
	def _judge_stopping(p):
		# p is the probability to go on
		return np.random.uniform() > p

	def _reject(self):
		# reject one bandit at least
		tmp_ucb_score = self._get_ucb_score()

		tmp_success_rate = {}
		for i in self.current_list:
			tmp_success_rate[i] = tmp_ucb_score[i]

		tmp_success_rate = self._min_max_scale(tmp_success_rate)
		print("REJECTING: tmp_success_Rate = {}".format(tmp_success_rate))
		self.current_list.clear()
		newly_rejected = 0

		while True:
			for i in tmp_success_rate.keys():
				# reject based on probability
				should_stop = self._judge_stopping(tmp_success_rate[i])
				if should_stop:
					# reject
					self.reject_list.add(i)
					newly_rejected += 1
				else:
					# go on
					self.current_list.add(i)
			# should reject at least one candidate, should leave at least one survivor
			if len(self.current_list) > 0 and len(self.reject_list) > 0:
				break

		print("Newly REJECTED {}/{}, LEAVE {}".format(newly_rejected, len(self.reject_list), len(self.current_list)))

	def _allocate_budgets(self, round_budget, budget_type):
		tmp_ucb = {}
		for i in self.current_list:
			tmp_ucb[i] = self.all_bandits[i].get_ucb_score() or 1
		tmp_ucb = self._min_max_scale(tmp_ucb)   # scale to assign higher ucb score to smaller loss
		pi = self._soft_max(tmp_ucb)
		for i in pi.keys():
			pi[i] = pi[i] * round_budget
			if budget_type == 'trial':
				pi[i] = int(pi[i])
		return pi

	@staticmethod
	def _min_max_scale(dic):
		min_val = min(dic.values())
		max_val = max(dic.values())
		res = {}
		if min_val == max_val:
			for i in dic.keys():
				res[i] = 1 / len(dic.keys())
		else:
			for i in dic.keys():
				res[i] = (max_val - dic[i]) / (max_val - min_val)
		return res

	@staticmethod
	def _soft_max(s):
		tmp_arr = list(s.values())
		tmp_arr = np.array(tmp_arr)
		tmp_arr = np.nan_to_num(tmp_arr)
		pi = {}
		for i in s.keys():
			pi[i] = np.exp(s[i]) / (np.sum(np.exp(tmp_arr)))
		return pi

	def _get_ucb_score(self):
		self.all_bandit_ucb_score = {}
		for k in self.current_list:
			self.all_bandit_ucb_score[k] = self.all_bandits[k].get_ucb_score() or 0
		return self.all_bandit_ucb_score

	def get_best_bandit(self):
		return self.best_bandit

	def get_key_of_best_bandit(self):
		return self.key_of_best_bandit

	def update_worst_loss(self, rewards):
		for loss in rewards:
			if self.worst_loss < loss:
				self.worst_loss = loss + 1

	def fit(self, X, y, fidelity_mb=None, random_state=None):

		if isinstance(fidelity_mb, (int, float)):
			X, y, X_size, X_fidelity_size = get_fidelity_input(X, y, mb=fidelity_mb, random_state=random_state)
			print("x_train to x_train_fidelity = {:.2f} MB to {:.2f} MB".format(X_size, X_fidelity_size))

		budget_per_round = int(self.total_budget / self.max_number_of_round)

		if budget_per_round <= 0.0:
			print("Budget per round = {}, not enough!".format(budget_per_round))
			return

		for i in range(self.max_number_of_round):

			if i == 0:
				for key in self.current_list:
					print("{}: {}, {}".format(key, self.all_bandits[key].algorithm_model.get_model_name(),
					                          self.all_bandits[key].get_best_record()))

			budget_for_every_bandit = self._allocate_budgets(budget_per_round, budget_type=self.budget_type)
			print(budget_for_every_bandit, "resource for every bandit")

			for k in self.current_list:
				print("Running {} ...".format(self.all_bandits[k].algorithm_model.get_model_name()))
				(rewards, model_parameters, out_learning_curve) = self.all_bandits[k].compute(
					X=X, y=y, X_val=None, y_val=None, budget=budget_for_every_bandit[k], budget_type=self.budget_type,
					validation_strategy='cv', validation_strategy_args=3, per_run_timebudget=self.per_run_timelimit,
					evaluation_rule=self.evaluation_rule, time_ref=self.time_ref, worst_loss=self.worst_loss,
					random_state=random_state)

				self.all_bandits[k].add_records(rewards, model_parameters)
				self.learning_curve.update(out_learning_curve)
				self.update_worst_loss(rewards)  # outer worst loss updater

			before_reject_current = len(self.current_list)
			self._reject()
			print("Round {}, from {} bandits to {} bandits".format(i, before_reject_current, len(self.current_list)))

	def propose_best_bandit_and_hyperparam(self):
		best_val_loss = None
		best_ind = -1
		best_para = None
		for i in self.all_bandits.keys():
			if best_val_loss is None or (self.all_bandits[i].get_best_record() is not None
			                             and best_val_loss > self.all_bandits[i].get_best_record()):
				best_val_loss = self.all_bandits[i].get_best_record()
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

	def predict(self, X):
		assert self.best_bandit is not None, "You should firstly refit, and then predict"
		y_prediction = self.best_bandit.algorithm_model.predict(X)
		return y_prediction

	def score(self, X, y):
		assert self.best_bandit is not None, "You should firstly refit, and then predict"
		y_prediction = self.best_bandit.algorithm_model.predict(X)

		print("Final Test Score: {}".format(eval_performance(self.evaluation_rule, y_true=y, y_score=y_prediction)))

	def print_summary(self):
		print("Total Actions: {}".format(sorted([x.get_num_of_actions() for x in self.all_bandits.values()],
		                                        reverse=True)))


class CompModelFilter(object):
	"""
	Pre-screening all composite models, run all of them one by one with default parameters.
	For saving time, we design a dynamic per_run_timelimit determination scheme.
	After the scores of all composite models are produced, we remove 80% bad composite models (followed by 80-20 rule).
	Then using USH to perform joint algorithm selection and hyperparameter optimization.
	"""
	def __init__(self, total_budget, budget_type, base_models=None, pickle_name="", time_ref=None,
	             minimum_run_timelimit=10.0, per_run_timelimit=240.0,
	             evaluation_rule='mean_squared_error', worst_loss=np.inf, random_state=None):
		self.total_budget = total_budget
		self.budget_type = budget_type
		self.per_run_timelimit = per_run_timelimit
		self.minimum_run_timelimit = minimum_run_timelimit

		self.base_models = base_models
		if base_models is None:
			self.base_models = list(get_all_regression_algorithm_keys())
			self.base_models.remove('FMRegressor')
			self.base_models.append('IdentityRegressor')

		self.evaluation_rule = evaluation_rule
		self.worst_loss = worst_loss
		self.random_state = random_state or 0

		self.algorithm_scores_1 = {}
		for algo1 in self.base_models:
			self.algorithm_scores_1[algo1] = 1.0
		self.algorithm_scores_2 = {}
		for algo2 in self.base_models:
			self.algorithm_scores_2[algo2] = 1.0

		self.gamma = 0.95
		self.pickle_name = pickle_name
		self.finished_composite_models = {}
		self.performance_dict = {}
		self.learning_curve = {}
		self.time_ref = time_ref or time.time()

	def get_learning_curve(self):
		return self.learning_curve

	@staticmethod
	def compute_process(X, y, in_comp_model, evaluation_rule='mean_squared_error', return_dict=None, random_state=0):
		comp_model = CompositeModel(
			[(in_comp_model[1], get_algorithm_by_key(in_comp_model[1], random_state=random_state)),
			 (in_comp_model[2], get_algorithm_by_key(in_comp_model[2], random_state=42 + random_state)),
			 in_comp_model[3]]
		)

		reward = comp_model.compute(X=X, y=y,
		                            evaluation_rule=evaluation_rule,
		                            validation_strategy_args=3,
		                            random_state=23 + random_state * 23)
		return_dict['val_{}'.format(evaluation_rule)] = reward['info']['val_{}'.format(evaluation_rule)]

	def selected_next_model(self, model_space: dict):
		sorted_model_space = sorted(model_space.items(),
		                            key=lambda x: self.algorithm_scores_1[x[0][1]] * self.algorithm_scores_2[x[0][2]],
		                            reverse=True)
		selected_model = sorted_model_space[0][0]
		return selected_model, self.algorithm_scores_1[selected_model[1]] * self.algorithm_scores_2[selected_model[2]]

	def construct_composite_model_space(self):
		composite_models_space = {}
		idx = 1
		for concat_type in ['c', 'p']:
			for algorithm1_key in self.base_models:
				for algorithm2_key in self.base_models:
					if algorithm2_key == 'IdentityRegressor':
						continue
					if algorithm1_key == 'IdentityRegressor':
						concat_type = 'o'
					composite_models_space[(idx, algorithm1_key, algorithm2_key, concat_type)] = 1.0
					idx += 1
		return composite_models_space

	def run(self, X, y, fidelity_mb=None):

		if isinstance(fidelity_mb, (int, float)):
			X, y, X_size, X_fidelity_size = get_fidelity_input(X, y, mb=fidelity_mb, random_state=self.random_state)
			print("x_train to x_train_fidelity = {:.2f} MB to {:.2f} MB".format(X_size, X_fidelity_size))

		total_memory = psutil.virtual_memory().total / 1048576.0
		minimum_free_memory = 1.0 / 16.0 * total_memory

		start_time = time.time()
		Time_Costs = [10, ]
		composite_model_space = self.construct_composite_model_space()
		num_composite_models = len(composite_model_space)
		first_model_score_less_1 = True

		print("Total Composite Models: {}".format(num_composite_models))

		while len(composite_model_space) > 0:
			comp_model, model_score = self.selected_next_model(composite_model_space)
			composite_model_space.pop(comp_model)
			free_memory = (psutil.virtual_memory().free + psutil.virtual_memory().cached) / 1048576.0
			if len(composite_model_space) % 10 == 0:
				print("Now Free Memory: {} MB".format(free_memory))
			if free_memory < minimum_free_memory:
				print("MemoryOut! Breaking...")
				break
			if first_model_score_less_1 and model_score < 1.0:
				first_model_score_less_1 = False
				print("{}/{} Model Score becomes <1.0,"
				      " Starting [mu+sigma] timeout !".format(len(self.finished_composite_models), num_composite_models))

			# Time Out should be user-defined self.minimum_run_timelimit(default 10) to
			# user-defined self.per_run_timelimit(default 360)
			if model_score < 1.0:  # all subsequent composite models have been punished, using [mu+sigma] timeout
				per_run_timelimit = max(self.minimum_run_timelimit, float(np.mean(Time_Costs) + np.std(Time_Costs)))
			else:                  # using mu+10*sigma to determine per_run_timelimit
				per_run_timelimit = (np.mean(Time_Costs) + np.std(Time_Costs)) * 10

			timeout = min(self.per_run_timelimit, float(per_run_timelimit))
			print("== Time Out: {:.1f} s".format(timeout))
			print(">> Running {}+{}+{}+{}".format(comp_model[0], comp_model[1], comp_model[2], comp_model[3]))
			inner_start_time = time.time()
			loss = self.worst_loss
			inner_time_cost = per_run_timelimit
			try:
				mgr = multiprocessing.Manager()
				return_dict = mgr.dict()
				p = multiprocessing.Process(target=self.compute_process, args=(X, y, comp_model, self.evaluation_rule,
				                                                               return_dict, self.random_state))
				p.start()

				if self.budget_type == 'time':
					p.join(min(self.total_budget-2-time.time()+start_time, per_run_timelimit))
				else:
					p.join(per_run_timelimit)

				if p.is_alive():
					p.terminate()
					kill_tree(p.pid)

				if return_dict['val_{}'.format(self.evaluation_rule)] is not None:
					val_score = return_dict['val_{}'.format(self.evaluation_rule)]
				else:
					val_score = loss_to_score(self.evaluation_rule, self.worst_loss)

				print("val_score {}".format(val_score))

				loss = score_to_loss(self.evaluation_rule, val_score)
				inner_time_cost = time.time()-inner_start_time
				Time_Costs.append(inner_time_cost)

			except Exception as e:
				print("Other Exceptions")
				print(e)
				loss = np.inf
				self.algorithm_scores_1[comp_model[1]] *= self.gamma
				self.algorithm_scores_2[comp_model[2]] *= self.gamma

			finally:
				if loss == self.worst_loss:
					inner_time_cost = time.time() - inner_start_time
				if self.budget_type == 'time' and time.time()-start_time >= self.total_budget-2.0:
					break

			print("{}+{}+{}+{}: val={}, TimeCost={}".format(comp_model[0], comp_model[1], comp_model[2], comp_model[3],
			                                                loss, inner_time_cost))
			self.performance_dict['{}+{}+{}+{}'.format(comp_model[0], comp_model[1],
			                                           comp_model[2], comp_model[3])] = (loss, inner_time_cost)
			self.learning_curve[time.time()-self.time_ref] = loss
			self.finished_composite_models[comp_model] = loss

		total_TimeCost = time.time()-start_time
		# self.performance_dict['Time_Cost'] = total_TimeCost
		print("TimeCost = {}".format(total_TimeCost))
		if self.pickle_name:
			pickle.dump(self.performance_dict, open('{}.pkl'.format(self.pickle_name), 'wb'))
		return total_TimeCost

	def get_bandits_and_worst_loss(self):
		sorted_mses = sorted(self.performance_dict.items(), key=lambda x: x[1][0], reverse=False)
		sorted_mses = sorted_mses[:max(1, int(0.2 * len(sorted_mses)))]  # 20-80 rule

		bandit_inst = {}
		idx = 0
		worst_loss = -np.inf
		for comp_model, (val_loss, time_cost) in sorted_mses:
			if val_loss != np.inf:
				m_idx, cm1, cm2, cat_type = comp_model.split('+')
				default_comp_model = CompositeModel([(cm1, get_algorithm_by_key(cm1, random_state=self.random_state)),
				                                     (cm2, get_algorithm_by_key(cm2, random_state=42 + self.random_state)),
				                                     cat_type])
				bandit_inst[idx] = Bandit(algorithm_model=default_comp_model)
				bandit_inst[idx].add_records([val_loss, ], [default_comp_model.get_params()])

				idx += 1
				# if worst loss not that worse
				if worst_loss < val_loss:
					worst_loss = val_loss + 1

		print("Total Bandits: {}".format(len(bandit_inst)))
		print("Worst Loss: {}".format(worst_loss))
		return bandit_inst, worst_loss


def kill_tree(pid, including_parent=True):
	parent = psutil.Process(pid)
	for child in parent.children(recursive=True):
		print("child", child)
		child.kill()

	if including_parent:
		parent.kill()


def get_worst_score_from_evaluation_rule(evaluation_rule):
	if evaluation_rule == 'mean_squared_error':
		return 1e90
	if evaluation_rule == 'accuracy_score':
		return 0.0
	raise NotImplementedError("Unsupported evaluation_rule Now...")


def get_worst_loss_from_evaluation_rule(evaluation_rule):
	if evaluation_rule == 'mean_squared_error':
		return 1e90
	if evaluation_rule == 'accuracy_score':
		return score_to_loss(evaluation_rule, 0.0)
	raise NotImplementedError("Unsupported evaluation_rule Now...")


def getmbof(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def sampling_fidelity(X, mb=1.0):
	low = 1
	high = X.shape[0]
	while low <= high:
		mid = (low + high) // 2
		if getmbof(X[:mid]) < mb:
			low = mid + 1
		else:
			high = mid - 1
	return high


def get_fidelity_input(X, y, mb=1.0, random_state=None):
	if getmbof(X) > mb:
		fidelity = sampling_fidelity(X=X, mb=mb)
		print("Proper Fidelity: {}".format(fidelity))
		np.random.seed(random_state)
		indexes = np.random.choice(len(X), fidelity)
		x_train_fidelity = X[indexes]
		y_train_fidelity = y[indexes]
	else:
		x_train_fidelity = X
		y_train_fidelity = y
	return x_train_fidelity, y_train_fidelity, getmbof(X), getmbof(x_train_fidelity)
