import os
import time
import das
import logging
import traceback
import numpy as np
import multiprocessing
from multiprocessing import Pool
from das.HypTuner.trial_scheduler.job import Job
from sklearn.model_selection import train_test_split
from das.crossvalidate import cross_validate_score
from das.performance_evaluation import score_to_loss


class TrialRunner(object):
	"""
	A TrialRunner is responsible for running a concrete hyper-parameter trial.
	For now, we use a process pool with just 1 process to be a proxy to run trials.
	Trials come from TrialScheduler.

	Parameters
	----------
	run_id
		string
		A unique identifier of that HyperBand run. Use, for example, the cluster's JobID when running multiple
		concurrent runs to separate them
	new_result_callback
		to be invoked when TrialRunner finished this trial, to register results
	estimator
		which estimator to run
	X
		input features
	y
		input labels
	validation_strategy
	validation_strategy_args
	logger
	verbose
	"""
	def __init__(self,
	             run_id='0',
	             new_result_callback=None,
	             estimator=None,
	             X=None,
	             y=None,
	             validation_strategy="cv",
	             validation_strategy_args=3,
	             logger=None,
	             verbose=True):
		self.run_id = run_id
		self.new_result_callback = new_result_callback
		self.process_pool = None
		self.logger = logger
		if self.logger is None:
			self.logger = logging.getLogger(das.logger_name)

		# computation stuff
		self.estimator = estimator
		# print("[TrialRunner] estimator = {}".format(estimator))
		self.config_space = estimator.get_config_space()
		self.config_transformer = self.config_space.get_config_transformer()
		self.X = X
		self.y = y
		self.validation_strategy = validation_strategy
		self.validation_strategy_args = validation_strategy_args
		self.verbose = verbose

	def create_pool(self, n_workers):
		self.process_pool = Pool(n_workers)

	def submit_job(self,
	               config_id,
	               timeout,
	               config,
	               budget_type,
	               budget_t,
	               working_directory,
	               **kwargs):
		"""
		Create a job to run.

		Parameters
		----------
		config_id
		timeout
		config
		budget_type
		budget_t
		working_directory
		kwargs

		Returns
		-------

		"""
		job = Job(cfg_id=config_id,
		          timeout=timeout,
		          config=config,
		          budget_type=budget_type,
		          budget_t=budget_t,
		          working_directory=working_directory,
		          **kwargs)
		job.time_it('submitted')
		self.run_job(job)

	def shutdown(self):
		if self.process_pool is not None:
			self.process_pool.close()
			self.process_pool.join()
		self.logger.debug("[TrialRunner] Shutdown Succeed!")

	def run_job(self, job):
		"""
		Run job

		Parameters
		----------
		job
			a Job instance, with related information of the job to be used on running process
		Returns
		-------

		"""
		if self.verbose:
			new_config = self.config_transformer(job.config, nick2ground=False)
			# print('start job {}: ({}, {})'.format(job.cfg_id, new_config, job.budget_t))
			self.logger.info(
				'start job {}: ({}, {})'.format(job.cfg_id, new_config, job.budget_t))
		job.time_it('started')
		try:
			result_future = self.process_pool.apply_async(computation, args=(job, ), callback=self.new_result_callback)
			# print("WAIT TIME: {}".format(job.timeout))
			job, reward = result_future.get(timeout=job.timeout)
			# print("JOB: {}".format(job))
			result = {'result': reward,
			          'exception': None}
		except multiprocessing.context.TimeoutError as e:
			# print("multiprocessing.context.TimeoutError")
			job.time_it('finished')
			reward = {'loss': score_to_loss(job.evaluation_rule, job.worst_score),
			          'info': {'val_{}'.format(job.evaluation_rule): job.worst_score,
			                   'exception': "TimeoutError: {}".format(e)}}
			job.result = reward
			job.exception = "TimeoutError"
			# print("JOB: {}".format(job))
			self.new_result_callback((job, None))
			result = {'result': reward,
			          'exception': "TimeoutError: {}".format(e)}
		except Exception as e:
			reward = {'loss': score_to_loss(job.evaluation_rule, job.worst_score),
			          'info': {'val_{}'.format(job.evaluation_rule): job.worst_score,
			                   'exception': "{}".format(e)}}
			print("run_job throw exception: {} -> {}".format(type(e), e))
			self.logger.debug("Computation Failed!\n{}".format(traceback.format_exc()))
			result = {'result': reward,
			          'exception': traceback.format_exc()}
		finally:
			# if result_future:
			# 	result_future.terminate()
			pass
		if self.verbose:
			self.logger.info(
				'job {} result: {}'.format(job.cfg_id, result))
		return result

	def async_run_job(self, job):
		# TODO: implement job running asynchronously
		self.logger.info('[TrialRunner] start processing job {}: ({}, {})'.format(job.cfg_id, job.config, job.budget_t))
		print('[TrialRunner] start job {}: ({}, {})'.format(job.cfg_id, job.config, job.budget_t))
		job.time_it('started')
		try:
			self.process_pool.apply_async(computation,
			                              args=(job, ),
			                              callback=self.new_result_callback)
		except multiprocessing.context.TimeoutError as e:
			job.time_it('finished')
			reward = {'loss': self.worst_reward, 'info': {'val_acc': 0.0, 'exception': "TimeoutError: {}".format(e)}}
			job.result = reward
			job.exception = "TimeoutError"
			# print("JOB: {}".format(job))
			self.new_result_callback((job, None))


def computation(job):
	"""
	Computation method, the method to be ran by the process pool

	Parameters
	----------
	job

	Returns
	-------

	"""
	exception = None
	try:
		reward = _compute(config_id=job.cfg_id,
		                  config=job.config,
		                  budget_type=job.budget_type,
		                  budget_t=job.budget_t,
		                  working_directory=job.working_directory,
		                  estimator=job.estimator,
		                  X=job.X,
		                  y=job.y,
		                  evaluation_rule=job.evaluation_rule,
		                  validation_strategy=job.validation_strategy,
		                  validation_strategy_args=job.validation_strategy_args,
		                  worst_score=job.worst_score,
		                  task=job.task,
		                  **job.kwargs)
	except Exception as e:
		print(e)
		reward = None
		exception = traceback.format_exc()
	job.time_it('finished')
	job.result = reward
	job.exception = exception
	return job, reward


def _compute(config_id,
             config=None,
             budget_type=None,
             budget_t=None,
             working_directory=".",
             estimator=None,
             X=None,
             y=None,
             evaluation_rule="accuracy_score",
             validation_strategy=None,
             validation_strategy_args=None,
             worst_score=None,
             task='classification',
             **kwargs):
		"""

		Parameters
		----------
		timeout: int
		    Time limit to compute
		config_id: tuple
			a triplet of ints that uniquely identifies a configuration. the convention is
			id = (iteration, budget index, running index) with the following meaning:
			- iteration: the iteration of the optimization algorithms. E.g, for HyperBand that is one round of
			 Successive Halving
			- budget index: the budget (of the current iteration) for which this configuration was sampled by the optimizer.
			 This is only nonzero if the majority of the runs fail and HyperBand re-samples to fill empty slots,
			  or you use a more 'advanced' optimizer.
			- running index: this is simply an int >= 0 that sort the configs into the order they where sampled,
			 i.e. (x,x,0) was sampled before (x,x,1).
		config: dict
			the actual configuration to be evaluated.
		budget: float
			the budget for the evaluation
		working_directory: str
			a name of a directory that is unique to this configuration. Use this to store intermediate results on
			 lower budgets that can be reused later for a larger budget (for iterative algorithms, for example).
		Returns
		-------
		dict:
			needs to return a dictionary with two mandatory entries:
				- 'loss': a numerical value that is MINIMIZED
				- 'info': This can be pretty much any build in python type, e.g. a dict with lists as value.
		"""
		# args and kwargs in BaseIteration.get_next_run() have only config_id, config and budget, we should
		# add X and y as dataset.

		budget, min_budget, max_budget = budget_t
		assert max_budget >= budget and max_budget > 0, "max_budget = {}, budget = {}, Wrong!".format(max_budget,
		                                                                                              budget)
		if budget_type == "datapoints":
			vol = int(len(X) * min(1, np.log(budget) / np.log(max_budget)))
			X = X[:vol]
			y = y[:vol]
		elif budget_type in ("time", "iter", "epoch"):
			X = X
			y = y
		else:
			raise RuntimeError("Cannot recognize budget_type {}".format(budget_type))

		worst_loss = score_to_loss(evaluation_rule, worst_score)

		if task in ('classification', 'regression'):
			if validation_strategy == "holdout":
				ratio_of_training_data = validation_strategy_args
				assert (0 <= ratio_of_training_data < 1), "无效的验证策略参数: {}".format(ratio_of_training_data)
				(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=1-ratio_of_training_data)
				try:
					reward = estimator.compute(config_id=config_id,
						                       config=config,
						                       budget_t=budget_t,
						                       X=X_train,
						                       y=y_train,
						                       X_val=X_valid,
						                       y_val=y_valid,
						                       evaluation_rule=evaluation_rule,
						                       working_directory=working_directory,
						                       task=task,
						                       **kwargs)
				except Exception as e:
					print(e)
					print(traceback.format_exc())
					reward = None

				if reward is None or 'loss' not in reward:
					reward = {'loss': worst_loss,
					          'info': {'val_{}'.format(evaluation_rule): worst_score,
					                   'exception': traceback.format_exc()}}

			elif validation_strategy == "cv":
				cv_fold = validation_strategy_args
				assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
				kwargs.update({'validation_strategy': 'cv',
				               'validation_strategy_args': cv_fold})
				try:
					reward = estimator.compute(config_id=config_id,
							                   config=config,
							                   budget_t=budget_t,
							                   X=X,
							                   y=y,
							                   X_val=None,
							                   y_val=None,
					                           evaluation_rule=evaluation_rule,
							                   working_directory=working_directory,
							                   task=task,
					                           **kwargs)
				except Exception as e:
					print(e)
					print(traceback.format_exc())
					reward = None
				if reward is None or 'loss' not in reward:
					reward = {'loss': worst_loss,
					          'info': {'val_{}'.format(evaluation_rule): worst_score,
					                   'exception': traceback.format_exc()}}
			else:
				raise Exception("无效验证策略: {}".format(kwargs['validation_strategy']))
		elif task == 'clustering':
			assert evaluation_rule is not None
			try:
				reward = estimator.compute(config_id=config_id,
						                   config=config,
						                   budget_t=budget_t,
						                   X=X,
						                   y=y,
						                   X_val=None,
						                   y_val=None,
				                           evaluation_rule=evaluation_rule,
						                   working_directory=working_directory,
				                           task=task,
				                           **kwargs)
			except Exception as e:
				print(e)
				print(traceback.format_exc())
				reward = None
			if reward is None or 'loss' not in reward:
				reward = {'loss': worst_loss,
				          'info': {'{}'.format(evaluation_rule): worst_score,
				                   'exception': traceback.format_exc()}}
		else:
			raise Exception("Invalid task: {}, should in (classification, clustering, regression)".format(task))
		return reward
