import das
import logging


class BaseConfigGenerator(object):
	"""
	The config generator determines how new configurations are sampled.
	This can take very different levels of complexity, from random sampling
	 to the construction of complex empirical prediction models for promising configurations.
	"""
	def __init__(self, config_space=None, logger=None):
		"""

		Parameters
		----------
		config_space
			ConfigSpace.ConfigurationSpace
			The configuration space to sample from. It contains the full
			specification of the Hyper-parameters with their priors
		logger
			logging.logger
		"""
		self.config_space = config_space
		self.logger = logger
		if logger is None:
			self.logger = logging.getLogger(das.logger_name)

	def get_config(self, budget=None):
		"""
		Function to sample a new configuration.

		This function is called inside Optimizer (e.g. HyperBand) to query a new configuration.

		Parameters
		----------
		budget: float
			the budget for which this configuration is scheduled
		Returns (config, info_dict)
			must return a valid configuration and a (possibly empty) info dict
		-------

		"""
		raise NotImplementedError('This function needs to be overwritten in %s.' % self.__class__.__name__)

	def new_result(self, job, update_model=True):
		"""
		Registers the result of finished runs.

		Every time a run has finished, this function should be called to register it with the result logger.
		If overwritten, make sure to call this method from the base class to ensure proper logging.

		Parameters
		----------
		job
			instance of dispatcher.Job
			contains all necessary information about the job
		update_model
			boolean
			determines whether a model inside the config_generator should be updated
		Returns
		-------

		"""
		if job.exception is not None:
			self.logger.warning("job {} failed with exception\n{}".format(job.cfg_id, job.exception))
