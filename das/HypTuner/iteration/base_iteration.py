import das
import copy
import logging
import numpy as np
from das.HypTuner.iteration.iteration_datum import Datum


class BaseIteration(object):
	"""
	Base class for various iteration possibilities.
	This decides what configuration should be run on what budget next.
	Typical choices are e.g. successive halving.
	Results from runs are processed and (depending on the implementations) determine the further development.

	Parameters
	----------

	cur_iter: int
		The current iteration index.
	num_configs: list of ints
		the number of configurations in each stage of SH
	budgets: list of floats
		the budget associated with each stage
	config_generator: das.HypTuner.config_gen.ConfigGenerator
		to produce config_getter: a function that returns a valid configuration. Its only
		argument should be the budget that this config is first
		scheduled for. This might be used to pick configurations
		that perform best after this particular budget is exhausted
		to build a better das system.
	logger: a logger
	result_logger: HypTuner.results.result.JsonResultLogger object
		a result logger that writes live results to the disk
	"""

	def __init__(self,
	             cur_iter,
	             num_configs,
	             budgets,
	             config_generator,
	             logger=None,
	             result_logger=None,
	             budget_type="time",
	             min_budget=None,
	             max_budget=None):
		self.data = {}  # this holds all the configs and results of this iteration
		self.is_finished = False
		self.cur_iter = cur_iter
		self.stage = 0  # internal iteration stage, but different name for clarity
		self.budgets = budgets
		self.budget_type = budget_type
		self.min_budget = min_budget
		self.max_budget = max_budget
		self.num_configs = num_configs
		self.actual_num_configs = [0] * len(num_configs)

		# make config_getter and config_transformer
		self.config_generator = config_generator
		self.config_getter = self.config_generator.get_config if self.config_generator is not None else None
		self.config_space = self.config_generator.config_space if self.config_generator is not None else None
		self.config_transformer = self.config_space.get_config_transformer() if self.config_space is not None else None

		self.num_running = 0
		self.logger = logger if logger is not None else logging.getLogger(das.logger_name)
		self.result_logger = result_logger

	def add_configuration(self, config=None, config_info=None):
		"""
		Function to add a new configuration to the current iteration

		Parameters
		----------
		config: a valid configuration
			The configuration to add. If None, a configuration is sampled from the config_sampler
		config_info: dict
			Some information about the configuration that will be stored in the results
		"""

		if config is None:
			config, config_info = self.config_getter(self.budgets[self.stage])  # ConfigGenerator.get_config(budget)
		# print("type(config) = {}, content = {}".format(type(config), config))
		# print("config_info = {}".format(config_info))
		if config_info is None:
			config_info = {}

		if self.is_finished:
			raise RuntimeError("This HypTune iteration is finished, you can't add more configurations!")

		if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
			raise RuntimeError("Can't add another configuration to stage %i"
			                   " in HypTune iteration %i." % (self.stage, self.cur_iter))

		config_id = (self.cur_iter, self.stage, self.actual_num_configs[self.stage])

		self.data[config_id] = Datum(config=config, config_info=config_info, budget=self.budgets[self.stage])

		self.actual_num_configs[self.stage] += 1

		if self.result_logger is not None:
			# MUST USE copy.deepcopy config, else config will be changed!!
			self.result_logger.new_config(config_id, copy.deepcopy(config), config_info)

		return config_id

	def register_result(self, job, skip_sanity_checks=False):
		"""
		Function to register the result of a job

		This function is called from the master of HypTuner, don't call this from your script.
		"""
		if self.is_finished:
			raise RuntimeError("This HypTune iteration is finished, you can't register more results!")

		config_id = job.cfg_id
		config = job.config
		# print("register result, config = {}".format(config))

		budget = job.budget_t[0]
		timestamps = job.timestamps
		result = job.result
		exception = job.exception

		d = self.data[config_id]  # Datum(config, config_info, budget)
		# print("register result, d.config = {}".format(d.config))
		if not skip_sanity_checks:
			assert_config_equal(d.config, config)
			assert d.status == 'RUNNING', "Configuration wasn't scheduled for a run. Status: {}".format(d.status)
			assert d.budget == budget, 'Budgets differ (%f != %f)!' % (self.data[config_id]['budget'], budget)

		d.time_stamps[budget] = timestamps
		d.results[budget] = result

		if (job.result is not None) and np.isfinite(result['loss']):
			d.status = 'REVIEW'
		else:
			d.status = 'CRASHED'  # job.result is None or loss is infinite

		d.exceptions[budget] = exception
		self.num_running -= 1

	def get_next_run(self):
		"""
		Function to return the next configuration and budget to run.

		This function is called from the master of HypTuner, don't call this from
		your script.

		It returns None if this run of SH is finished or there are
		pending jobs that need to finish to progress to the next stage.

		If there are empty slots to be filled in the current SH stage
		(which never happens in the original SH version), a new
		configuration will be sampled and scheduled to run next.
		"""

		if self.is_finished:
			return None

		for k, v in self.data.items():
			if v.status == 'QUEUED':   # we just perform the queued job
				assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
				v.status = 'RUNNING'
				self.num_running += 1
				# print("[base_iteration] self.config_transformer = {}".format(self.config_transformer))
				# print("[base_iteration]v.config = {}".format(v.config))
				# print("[base_iteration]convert(v.config) = ", self.config_transformer(v.config, nick2ground=True))
				return (k,
				        # we transform the config from all string to TRUE config which may contains objects, functions..
				        v.config if self.config_transformer is None \
					       else self.config_transformer(v.config, nick2ground=True),  # transform
				        (v.budget, self.min_budget, self.max_budget))

		# check if there are still slots to fill in the current stage and return that
		if self.actual_num_configs[self.stage] < self.num_configs[self.stage]:
			self.add_configuration()
			return self.get_next_run()

		if self.num_running == 0:
			# at this point a stage is completed
			self.process_results()
			return self.get_next_run()

		return None

	def _advance_to_next_stage(self, config_ids, losses):
		"""
		Function that implements the strategy to advance configs within this iteration

		Overload this to implement different strategies, like
		SuccessiveHalving, SuccessiveResampling.

		Parameters
		----------
		config_ids: list
			all config ids to be considered
		losses: numpy.array
			losses of the run on the current budget

		Returns
		-------
		list of bool
			A boolean for each entry in config_ids indicating whether to advance it or not
		"""
		raise NotImplementedError('_advance_to_next_stage not implemented for %s' % type(self).__name__)

	def process_results(self):
		"""
		Function that is called when a stage is completed and needs to be analyzed before further computations.

		The code here implements the original SH algorithms by advancing the k-best (lowest loss) configurations
		 at the current budget. k is defined by the num_configs list (see __init__) and the current stage value.

		For more advanced methods like resampling after each stage, overload this function only.
		"""
		self.stage += 1

		# collect all config_ids that need to be compared
		config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))

		# stage arrives the end, finish up
		if self.stage >= len(self.num_configs):
			self.finish_up()
			return

		budgets = [self.data[cid].budget for cid in config_ids]
		if len(set(budgets)) > 1:
			raise RuntimeError('Not all configurations have the same budget!')

		budget = self.budgets[self.stage - 1]

		losses = np.array([self.data[cid].results[budget]['loss'] for cid in config_ids])

		advance = self._advance_to_next_stage(config_ids, losses)

		for i, a in enumerate(advance):
			if a:
				self.logger.debug('ITERATION: Advancing config %s to'
				                  ' next budget %f' % (config_ids[i], self.budgets[self.stage]))

		for i, cid in enumerate(config_ids):
			if advance[i]:
				self.data[cid].status = 'QUEUED'
				self.data[cid].budget = self.budgets[self.stage]
				self.actual_num_configs[self.stage] += 1
			else:
				self.data[cid].status = 'TERMINATED'

	def finish_up(self):
		self.is_finished = True

		for k, v in self.data.items():
			assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finished yet!'
			v.status = 'COMPLETED'

	def __repr__(self):
		return ("\nIteration:\n" +
		        "\tcur_iter = {},\n".format(self.cur_iter) +
		        "\tstage = {}\n".format(self.stage) +
		        "\tbudgets = {}\n".format(self.budgets) +
		        "\tbudget_type, min_budget, max_budget = {}, {}, {}\n".format(self.budget_type, self.min_budget,
		                                                                      self.max_budget) +
		        "\tnum_configs = {}\n".format(self.num_configs) +
		        "\tconfig_sampler = {}\n".format(self.config_getter.__class__))


def assert_config_equal(config_1, config_2):
	"""
	Assert config_1 == config_2.
	Ignore the object non-equal.

	Parameters
	----------
	config_1
	config_2

	Returns
	-------

	"""
	assert config_1.keys() == config_2.keys(), "The key of two configs are unequal"
	for key in config_1.keys():
		if not isinstance(config_1[key], (float, int, str, bool, list, tuple, set, dict)):
			continue
		else:
			assert config_1[key] == config_2[key], ("Configuration differ. config_1[{}]={} but config_2[{}]={}".format(
				key, config_1[key], key, config_2[key]
			))
	return
