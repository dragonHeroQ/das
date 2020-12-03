import os
import time
import copy
import das
import logging
import threading
from das.HypTuner.result import Result
from das.HypTuner.trial_runner.trial_runner import TrialRunner
from das.performance_evaluation import loss_to_score, compare_and_update
from das.HypTuner.iteration.warmstart_iteration import WarmStartIteration


class TrialScheduler(object):
	"""
	The TrialScheduler class is responsible for the book keeping and to decide what to run next.
	For example, RandomSearch, HyperBand, and BOHB, that handle the important steps of deciding
	 what configurations to run on what budget when.

	Parameters
	----------
	run_id
		string
		A unique identifier of that HyperBand run. Use, for example, the cluster's JobID when running multiple
		concurrent runs to separate them
	config_generator
		das.HypTuner.config_gen object
		An object that can generate new configurations and registers results of executed runs
	budget_type
	estimator
	X
	y
	evaluation_rule
	validation_strategy
	validation_strategy_args
	logger
		logging.logger like object
		The logger to output some (more or less meaningful) information
	result_logger
		HypTuner.result.json_result_logger object
		a result logger that writes live results to disk
	working_directory
		string
		The top level working directory accessible to all compute nodes(shared filesystem).
	warm_start_result
		warm start results to be used
	verbose
	worst_score
	task
	"""
	def __init__(self,
	             run_id,
	             config_generator,
	             budget_type=None,
	             estimator=None,
	             X=None,
	             y=None,
	             evaluation_rule="accuracy_score",
	             validation_strategy=None,
	             validation_strategy_args=None,
	             logger=None,
	             result_logger=None,
	             working_directory=".",
	             warm_start_result=None,
	             verbose=True,
	             worst_score=None,
	             task=None):
		self.run_id = run_id
		self.config_generator = config_generator
		self.budget_type = budget_type
		assert self.budget_type in ('time', 'epoch', 'datapoints', 'iter'), ("UNKNOWN budget type: {}".format(budget_type))

		self.logger = logger
		if self.logger is None:
			self.logger = logging.getLogger(das.logger_name)

		self.result_logger = result_logger
		self.working_directory = working_directory
		os.makedirs(self.working_directory, exist_ok=True)

		self.time_ref = None
		self.iterations = []
		self.jobs = []
		self.num_running_jobs = 0

		# warm start stuff
		self.warm_start_result = warm_start_result
		if self.warm_start_result is None:
			self.warm_start_iterations = []
		else:
			self.warm_start_iterations = [WarmStartIteration(self.warm_start_result, self.config_generator), ]

		self.worst_score = worst_score
		self.task = task
		assert self.task is not None, ("self.task == None, please assign a valid task"
		                               " (classification, regression or clustering)")

		# condition to synchronize the job_callback
		self.thread_cond = threading.Condition()

		# Runner stuff
		self.estimator = estimator
		self.X = X
		self.y = y
		self.evaluation_rule = evaluation_rule    # for clustering
		self.validation_strategy = validation_strategy
		self.validation_strategy_args = validation_strategy_args
		self.sched_config = {'time_ref': self.time_ref,
		                     'evaluation_rule': self.evaluation_rule}
		self.running_kwds = {'estimator': self.estimator,
		                     'X': self.X,
		                     'y': self.y,
		                     'evaluation_rule': self.evaluation_rule,
		                     'validation_strategy': self.validation_strategy,
		                     'validation_strategy_args': self.validation_strategy_args,
		                     'worst_score': self.worst_score,
		                     'task': self.task}

		# TrialRunner to run trials
		self.runner = TrialRunner(run_id=run_id,
		                          new_result_callback=self.job_callback,
		                          estimator=self.estimator,
		                          verbose=verbose)

	def get_next_iteration(self, iteration, iteration_kwargs):
		"""
		Instantiates the next iteration

		Overwrite this to change the iterations for different schedulers

		Parameters
		----------
		iteration: int
			the index of the iteration to be instantiated
		iteration_kwargs: dict
			additional kwargs for the iteration class

		Returns
		-------
		iteration: das.HypTuner.iteration object
			a valid iteration object
		"""
		raise NotImplementedError('implement get_next_iteration for your scheduler %s' % type(self).__name__)

	def shutdown(self):
		self.logger.debug('TrialScheduler: shutdown initiated')
		self.runner.shutdown()

	def run(self,
	        total_time_budget=3600,
	        per_run_time_limit=360,
	        n_iterations=1e9,
	        n_workers=1,
	        iteration_kwargs=None):
		"""
		Run n_iterations, e.g. SuccessiveHalving or SuccessiveResampling.

		Parameters
		----------
		n_iterations: int
			number of iterations to be performed in this run
		n_workers: int
			the number of workers before starting the run
		total_time_budget: int
		    total time budget for this run task
		per_run_time_limit: int
		    time limit for one run of a configuration
		iteration_kwargs: dict
		"""

		if self.time_ref is None:
			self.time_ref = time.time()
			self.sched_config['time_ref'] = self.time_ref
		self.logger.info('starting run at %s' % (str(self.time_ref)))

		self.runner.create_pool(n_workers)
		if iteration_kwargs is None:
			iteration_kwargs = {}
		iteration_kwargs.update({'result_logger': self.result_logger})

		self.thread_cond.acquire()
		# Scheduler loop
		while True:
			# print("WORST_SCORE: {}".format(self.worst_score))
			time_1 = time.time()
			if time_1 - self.time_ref >= total_time_budget - 1.5:
				break

			next_run = None
			# find a new run to schedule
			idx = 0
			for i in self.active_iterations():
				next_run = self.iterations[i].get_next_run()
				idx = i
				if next_run is not None:
					break

			time_2 = time.time()
			time_left = total_time_budget - 1.5 - (time_2 - self.time_ref)
			if time_left <= 1:  # only 1.5 second, exit
				break

			if time_left > per_run_time_limit:  # if time_left is enough, leave per_run_time_limit as timeout
				timeout = per_run_time_limit
			else:                               # else leave time_left-1.5 as timeout
				timeout = time_left - 1.5

			if timeout < 1.0:
				break

			if next_run is not None:
				next_run += (timeout,)
				# print("next_RUN: ", next_run)
				self.logger.debug('schedule new run for iteration %i' % idx)
				self._submit_job(*next_run, **self.running_kwds)
				continue
			else:
				if n_iterations > 0:  # we might be able to start the next iteration
					self.iterations.append(self.get_next_iteration(len(self.iterations), iteration_kwargs))
					n_iterations -= 1
					self.logger.info("Iteration {} {}".format(len(self.iterations), "============>"))
					continue
			# not continue
			# at this point there is no immediate run that can be scheduled,
			# so wait for some job to finish if there are active iterations
			if self.active_iterations():
				self.thread_cond.wait()
			else:
				break

		self.thread_cond.release()

		for it in self.warm_start_iterations:
			it.fix_timestamps(self.time_ref)
		#
		ws_data = [i.data for i in self.warm_start_iterations]

		return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.sched_config)

	def active_iterations(self):
		"""
		Function to find active (not marked as finished) iterations

		Returns
		-------
			list: all active iteration objects (empty if there are none)
		"""

		active_iters = list(filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))))
		return active_iters

	def _submit_job(self, config_id, config, budget_t, timeout, **kwargs):
		"""
		Hidden function to submit a new job to the runner

		This function handles the actual submission in a (hopefully) thread save way

		Parameters
		----------
		config_id
		config
		budget_t
		timeout
		kwargs

		Returns
		-------

		"""
		self.logger.debug('trying submitting job %s to runner' % str(config_id))

		# with self.thread_cond:
		self.logger.debug('submitting job %s to runner' % str(config_id))
		# print("[TrialScheduler] submit job config = {}".format(config))
		self.runner.submit_job(config_id=config_id,
		                       timeout=timeout,
		                       config=config,
		                       budget_type=self.budget_type,
		                       budget_t=budget_t,
		                       working_directory=self.working_directory,
		                       **kwargs)
		self.num_running_jobs += 1

		# shouldn't the next line be executed while holding the condition?
		self.logger.debug("job %s submitted to runner" % str(config_id))

	def job_callback(self, job_reward):
		"""
		Method to be called when a job has finished

		This will do some book keeping and call the user defined new_result_callback if one was specified

		Parameters
		----------
		job_reward
			tuple
			trial_scheduler.Job object and reward: returned by process
		Returns
		-------

		"""
		job, reward = job_reward
		# self.logger.debug('[TrialScheduler] job_callback for %s started' % str(job.cfg_id))
		# print('[TrialScheduler] job_callback for %s started' % str(job.cfg_id))
		# print("JOB result: {}".format(job.result))
		# with self.thread_cond:
		try:
			# self.logger.debug('[TrialScheduler] job_callback for %s got condition' % str(job.cfg_id))
			# print('[TrialScheduler] job_callback for %s got condition' % str(job.cfg_id))
			self.num_running_jobs -= 1

			if self.result_logger is not None:
				self.result_logger(job)
			# job.cfg_id[0] is cur_iter
			self.iterations[job.cfg_id[0]].register_result(job)
			self.config_generator.new_result(job)
			self.worst_score_callback(job)
		except Exception as e:
			print(e)

		self.logger.debug('job_callback for %s finished' % str(job.cfg_id))
		# print('[TrialScheduler] job_callback for %s finished' % str(job.cfg_id))

	def worst_score_callback(self, job):
		score = loss_to_score(job.evaluation_rule, job.result['loss'])
		self.worst_score = compare_and_update(job.evaluation_rule, self.worst_score, score)

