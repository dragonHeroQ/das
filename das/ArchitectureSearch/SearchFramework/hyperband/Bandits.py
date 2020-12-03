import das
import time
import logging
import numpy as np
import multiprocessing
from das.util.common_utils import kill_tree
from das.crossvalidate import cross_validate_score
from das.BaseAlgorithm.BaseEstimator import BaseEstimator
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool
from das.ArchitectureSearch.Optimizer.RandomSearchOptimizer import RandomSearchOptimizer
from das.performance_evaluation import eval_performance, loss_to_score, score_to_loss

logger = logging.getLogger(das.logger_name)


class Bandit(object):

	def __init__(self,
	             learning_tool: BaseLearningTool=None,
	             algorithm_model: BaseEstimator=None,
	             c=2, ucb_mean=0, ucb_var=1e-5):

		self.learning_tool = learning_tool
		self.algorithm_model = algorithm_model
		self.algorithm_model = self.learning_tool.learning_estimator

		# records
		self.records = []
		self.loss_records = []
		self.model_parameters = []

		# mean and variation
		self.u = ucb_mean
		self.v = ucb_var

		# constant to control gaussian ucb
		self.c = c
		self.num_of_actions = 0
		self.reward = None

	def get_mean(self):
		if len(self.loss_records) > 0:
			self.u = np.mean(self.loss_records)
		return self.u

	def get_variation(self):
		if len(self.loss_records) > 0:
			self.v = np.std(self.loss_records)
		return self.v

	def get_num_of_actions(self):
		return self.num_of_actions

	def _gaussian_ucb_score(self):
		# gaussian ucb
		import math
		tmp_score = self.get_mean() + self.c * self.get_variation() / math.log(self.num_of_actions + 2, 2)
		return tmp_score

	def _best_score(self):
		if len(self.loss_records) == 0:
			return None
		# get the record with the minimum loss
		return min(self.loss_records)

	def get_ucb_score(self):
		# return self._best_record()
		return self._gaussian_ucb_score()

	def add_records(self, model_parameters=None, in_records=None):
		if (not model_parameters) or (not in_records):  # nothing to add
			return
		assert len(model_parameters) == len(in_records), "number of model_parameters != number of result records"
		self.model_parameters.extend(model_parameters)
		self.records.extend(in_records)
		self.loss_records.extend(list(map(lambda x: x['loss'], in_records)))
		self.num_of_actions += len(in_records)

	@staticmethod
	def cv_score(model, X, y, cv_fold, evaluation_rule, return_dict, random_state=None):
		return_dict["val_score"] = None
		val_score, _ = cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule,
		                                    random_state=random_state)
		# return_dict["model"] = model
		return_dict["val_score"] = val_score

	@staticmethod
	def holdout_score(model, X, y, X_val, y_val, evaluation_rule, return_dict, random_state=None):
		return_dict["val_score"] = None
		model.fit(X, y)
		y_hat = model.predict(X_val)
		val_score = eval_performance(rule=evaluation_rule, y_true=y_val, y_score=y_hat,
		                             random_state=random_state)
		# return_dict["model"] = model
		# Note: currently no need to store and return model object
		return_dict["val_score"] = val_score

	def compute_deep_archi(self, evaluator, X, y, budget=3, budget_type='trial',
	                       per_run_timelimit=240.0, random_state=None, **kwargs):
		res = []
		model_parameters = []
		tmp_start_time = time.time()
		num_trials = 0

		while True:
			if ((budget_type == 'time' and time.time() - tmp_start_time >= budget - 2)
				  or (budget_type == 'trial' and num_trials >= budget)):  # stopping criteria
				break
			tps = self.learning_tool.learning_estimator.get_configuration_space()
			rs = RandomSearchOptimizer(tps)
			candidate_config = rs.get_random_config()
			# debug at 01/22 15:37
			# candidate_config = {'0/0/block0/0#0/N/QuadraticDiscriminantAnalysis/reg_param': 0.21037100935034458,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/n_estimators': 474,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/criterion': 'entropy',
			#                     '0/1/block1/0#1/N/RandomForestClassifier/max_depth': 77,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/max_features': 1.0,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/min_samples_split': 19,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/min_samples_leaf': 13,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/bootstrap': True,
			#                     '0/1/block1/0#1/N/RandomForestClassifier/oob_score': False}
			logger.info("Start Config: {}".format(candidate_config))
			model_parameters.append(candidate_config)
			self.learning_tool.learning_estimator.set_params(**candidate_config)

			if budget_type == 'time':
				run_time_limit = budget - (time.time() - tmp_start_time)
			else:
				run_time_limit = per_run_timelimit

			reward = evaluator.evaluate(learning_tool=self.learning_tool, X=X, y=y,
			                            run_time_limit=run_time_limit, random_state=random_state, **kwargs)
			print("Reward = {}".format(reward))
			res.append(reward)

			num_trials += 1

		return model_parameters, res

	# def compute(self, X, y, X_val=None, y_val=None, budget=3, budget_type="trial",
	#             validation_strategy='cv', validation_strategy_args=3, per_run_timebudget=360.0,
	#             evaluation_rule=None, time_ref=None, n_classes=None, worst_loss=None, random_state=None):
	#
	# 	model = self.algorithm_model
	# 	assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"
	#
	# 	res = []
	# 	model_parameters = []
	# 	tmp_start_time = time.time()
	# 	num_trials = 0
	# 	if time_ref is None:
	# 		time_ref = time.time()
	# 	learning_curve = {}
	#
	# 	while True:
	# 		failed_flag = False
	# 		tps = self.algorithm_model.get_configuration_space()
	# 		rs = RandomSearchOptimizer(tps)
	# 		candidate_config = rs.get_random_config()
	# 		print("Starting Config: {}".format(candidate_config))
	# 		model_parameters.append(candidate_config)
	# 		self.algorithm_model.set_params(**candidate_config)
	#
	# 		mgr = multiprocessing.Manager()
	# 		return_dict = mgr.dict()
	#
	# 		try:
	# 			if X_val is not None:  # holdout
	# 				assert validation_strategy == 'holdout', 'Validation Strategy is not holdout, why X_val provided?'
	# 				p = multiprocessing.Process(target=self.holdout_score, args=(model, X, y, X_val, y_val,
	# 				                                                             evaluation_rule, return_dict,
	# 				                                                             random_state))
	# 			else:
	# 				assert validation_strategy == 'cv', 'Validation Strategy should be cv if X_val is not provided'
	# 				cv_fold = validation_strategy_args
	# 				assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
	# 				p = multiprocessing.Process(target=self.cv_score, args=(model, X, y, cv_fold, evaluation_rule,
	# 				                                                        return_dict, random_state))
	# 			p.start()
	# 			if budget_type == 'time':
	# 				per_runtime_limit = budget - (time.time() - tmp_start_time)
	# 			else:
	# 				per_runtime_limit = per_run_timebudget
	# 			print("Time Limit: {}".format(per_runtime_limit))
	# 			p.join(per_runtime_limit)
	#
	# 			if p.is_alive():
	# 				p.terminate()
	# 				kill_tree(p.pid)
	#
	# 			if 'val_score' in return_dict and return_dict["val_score"] is not None:
	# 				val_score = return_dict["val_score"]
	# 			else:
	# 				failed_flag = True  # exception occurred
	# 				val_score = loss_to_score(evaluation_rule, worst_loss)
	#
	# 			print("val_score {}".format(val_score))
	#
	# 			loss = score_to_loss(evaluation_rule, val_score)
	# 			self.reward = {'loss': loss,
	# 			               'info': {'val_{}'.format(evaluation_rule): val_score}}
	# 			if worst_loss < loss:  # local worst loss update
	# 				worst_loss = loss + 1
	# 		except Exception as e:
	# 			print(e)
	# 			failed_flag = True
	# 			self.reward = {'loss': worst_loss,
	# 			               'info': {'val_{}'.format(evaluation_rule): loss_to_score(evaluation_rule, worst_loss)}}
	# 		finally:
	# 			learning_curve[time.time() - time_ref] = self.reward['loss']
	# 			res.append(self.reward['loss'])
	# 			if failed_flag:
	# 				print("Exception Occurred!")
	# 		num_trials += 1
	#
	# 		if ((budget_type == 'time' and time.time() - tmp_start_time >= budget - 2)
	# 			  or (budget_type == 'trial' and num_trials >= budget)):  # stopping criteria
	# 			break
	#
	# 	return model_parameters, res, learning_curve

	def get_best_score(self):
		return self._best_score()

	def get_best_model_parameters(self):
		if len(self.model_parameters) < 1 or len(self.loss_records) < 1:
			return None
		ind = int(np.argmin(np.array(self.loss_records)))
		return self.model_parameters[ind]
