import traceback
import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from das.HypTuner.config_gen.base_config_generator import BaseConfigGenerator


class BOHBConfigGenerator(BaseConfigGenerator):
	"""
	Fits for each given budget a kernel density estimator on the best N percent of the
	evaluated configurations on this budget.

	Parameters:
	-----------
	config_space: ConfigSpace
		Configuration space object
	top_n_percent: int
		Determines the percentile of configurations that will be used as training data
		for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
		for training.
	min_points_in_model: int
		minimum number of data-points needed to fit a model
	num_samples: int
		number of samples drawn to optimize EI via sampling
	random_fraction: float
		fraction of random configurations returned
	bandwidth_factor: float
		widens the bandwidth for continuous parameters for proposed points to optimize EI
	min_bandwidth: float
		to keep diversity, even when all (good) samples have the same value for one of the parameters,
		a minimum bandwidth (Default: 1e-3) is used instead of zero.

	"""
	def __init__(self,
	             config_space,
	             min_points_in_model=None,
	             top_n_percent=15,
	             num_samples=64,
	             random_fraction=1 / 3,
	             bandwidth_factor=3,
	             min_bandwidth=1e-3,
	             **kwargs):
		super().__init__(config_space=config_space, **kwargs)
		self.top_n_percent = top_n_percent
		self.config_space = config_space
		self.config_transformer = self.config_space.get_config_transformer()
		self.bw_factor = bandwidth_factor
		self.min_bandwidth = min_bandwidth

		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:  # default min points in model is dim+1
			self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

		if self.min_points_in_model < len(self.config_space.get_hyperparameters()) + 1:
			self.logger.warning('Invalid min_points_in_model value.'
			                    ' Setting it to %i' % (len(self.config_space.get_hyperparameters()) + 1))
			self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

		self.num_samples = num_samples
		self.random_fraction = random_fraction

		hyper_parameters = self.config_space.get_hyperparameters()

		self.kde_vartypes = ""
		self.vartypes = []

		for hyp in hyper_parameters:
			if hasattr(hyp, 'sequence'):
				raise RuntimeError('This version on HypTuner does not support ordinal hyper-parameters.'
				                   ' Please encode %s as an integer parameter!' % hyp.name)
			if hasattr(hyp, 'choices'):
				self.kde_vartypes += 'u'
				self.vartypes += [len(hyp.choices)]
			else:
				self.kde_vartypes += 'c'
				self.vartypes += [0]

		self.vartypes = np.array(self.vartypes, dtype=int)

		# store precomputed probabilities for the categorical parameters
		self.cat_probs = []

		self.configs = dict()
		self.losses = dict()
		self.good_config_rankings = dict()
		self.kde_models = dict()

	def largest_budget_with_model(self):
		if len(self.kde_models) == 0:
			return -float('inf')
		return max(self.kde_models.keys())

	def get_config(self, budget=None):
		"""
		Function to sample a new configuration.

		This function is called inside Optimizer (e.g. HyperBand) to query a new configuration.

		Parameters
		----------
		budget
			float
			the budget for which this configuration is scheduled

		Returns
			(config, info_dict)
			must return a valid configuration and a (possibly empty) info dict
		-------

		"""
		self.logger.debug('start sampling a new configuration.')

		sample = None
		info_dict = {}

		# If no model is available, sample from prior
		# also mix in a fraction of random configs
		if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
			sample = self.config_space.sample_configuration()
			info_dict['model_based_pick'] = False

		best = np.inf
		best_vector = None

		if sample is None:
			try:
				# sample from largest budget
				budget = max(self.kde_models.keys())

				l = self.kde_models[budget]['good'].pdf
				g = self.kde_models[budget]['bad'].pdf

				def minimize_me(x):
					return max(1e-32, g(x)) / max(l(x), 1e-32)

				kde_good = self.kde_models[budget]['good']
				kde_bad = self.kde_models[budget]['bad']

				for i in range(self.num_samples):
					idx = np.random.randint(0, len(kde_good.data))
					datum = kde_good.data[idx]
					vector = []

					for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

						bw = max(bw, self.min_bandwidth)
						if t == 0:
							bw = self.bw_factor * bw
							try:
								vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
							except:
								self.logger.warning("Truncated Normal failed for:\ndatum=%s\n"
								                    "bandwidth=%s\nfor entry with value %s" % (datum, kde_good.bw, m))
								self.logger.warning("data in the KDE:\n%s" % kde_good.data)
						else:
							if np.random.rand() < (1 - bw):
								vector.append(int(m))
							else:
								vector.append(np.random.randint(t))
					val = minimize_me(vector)

					if not np.isfinite(val):  # val is inf
						self.logger.debug('sampled vector: %s has EI value %s' % (vector, val))
						self.logger.debug("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
						self.logger.debug("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
						self.logger.warning("l(x) = %s" % (l(vector)))
						self.logger.warning("g(x) = %s" % (g(vector)))

						# right now, this happens because a KDE does not contain all values for a categorical parameter
						# this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
						# if the good_kde has a finite value, i.e. there is no config with that value in the bad kde,
						# so it shouldn't be terrible.
						if np.isfinite(l(vector)):
							best_vector = vector
							break

					if val < best:
						best = val
						best_vector = vector

				if best_vector is None:
					# Sampling based optimization with samples failed
					self.logger.debug("[BEST_VECTOR_NONE] Sampling based optimization with %i samples failed"
					                  " -> using random configuration" % self.num_samples)
					sample = self.config_space.sample_configuration().get_dictionary()
					info_dict['model_based_pick'] = False
				else:
					# Sampling based optimization with samples succeed
					self.logger.debug(
						'best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
					for i, hp_value in enumerate(best_vector):
						if isinstance(
								self.config_space.get_hyperparameter(
									self.config_space.get_hyperparameter_by_idx(i)
								),
								ConfigSpace.hyperparameters.CategoricalHyperparameter
						):
							best_vector[i] = int(np.rint(best_vector[i]))
					sample = ConfigSpace.Configuration(self.config_space,
					                                   vector=np.array(best_vector))  # .get_dictionary()
					info_dict['model_based_pick'] = True
			except:
				self.logger.warning("[EXCEPTION] Sampling based optimization with %i samples failed\n %s \n"
				                    "Using random configuration" % (self.num_samples, traceback.format_exc()))
				sample = self.config_space.sample_configuration()
				info_dict['model_based_pick'] = False

		# print("Config Space is {}".format(self.config_space))
		# print("[Before Deactivate] Sample: ({}){}".format(type(sample), sample.get_dictionary()))
		try:
			sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
				configuration_space=self.config_space,
				configuration=sample.get_dictionary()
			).get_dictionary()
			# print("[Before Deactivate] Sample: ({}){}".format(type(sample), sample.get_dictionary()))
		except Exception as e:
			self.logger.warning("Error (%s) converting configuration: %s -> "
			                    "using random configuration!",
			                    e,
			                    sample)
			sample = self.config_space.sample_configuration().get_dictionary()
		self.logger.debug('done sampling a new configuration.')

		return sample, info_dict

	def impute_conditional_data(self, array):

		return_array = np.empty_like(array)

		for i in range(array.shape[0]):
			datum = np.copy(array[i])
			nan_indices = np.argwhere(np.isnan(datum)).flatten()

			while np.any(nan_indices):
				nan_idx = nan_indices[0]
				valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

				if len(valid_indices) > 0:
					# pick one of them at random and overwrite all NaN values
					row_idx = np.random.choice(valid_indices)
					datum[nan_indices] = array[row_idx, nan_indices]

				else:
					# no good point in the data has this value activated, so fill it with a valid but random value
					t = self.vartypes[nan_idx]
					if t == 0:
						datum[nan_idx] = np.random.rand()
					else:
						datum[nan_idx] = np.random.randint(t)

				nan_indices = np.argwhere(np.isnan(datum)).flatten()
			return_array[i, :] = datum
		return return_array

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
		super().new_result(job)

		if job.result is None:
			# One could skip crashed results, but we decided
			# assign a +inf loss and count them as bad configurations
			loss = np.inf
		else:
			loss = job.result["loss"]

		budget = job.budget_t[0]

		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []

		# skip model building if we already have a bigger model
		if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
			return

		# We want to get a transformed representation of the configuration in the original space
		# print("[BOHB_CONFIG_GEN] before", job.config)
		config = job.config
		# print(job.config)
		if self.config_transformer is not None:
			# print("[BOHB_CONFIG_GEN]", self.config_transformer)
			config = self.config_transformer(job.config, nick2ground=False)
		# print("[BOHB_CONFIG_GEN] job.config = {}".format(job.config))
		# print("[bohb_config_gen] self.config_space = {}".format(self.config_space))
		conf = ConfigSpace.Configuration(self.config_space, config)
		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(loss)

		# skip model building:
		# a) if not enough points are available
		if len(self.configs[budget]) < self.min_points_in_model:
			self.logger.debug("Only %i run(s) for budget %f available,"
			                  " need more than %s -> can't build model!" % (len(self.configs[budget]), budget,
			                                                                self.min_points_in_model + 1))
			return

		# b) during warm starting when we feed previous results in and only update once
		if not update_model:
			return

		train_configs = np.array(self.configs[budget])
		train_losses = np.array(self.losses[budget])

		n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
		n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

		# Refit KDE for the current budget
		idx = np.argsort(train_losses)

		train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
		train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good + n_bad]])

		if train_data_good.shape[0] <= train_data_good.shape[1]:
			return
		if train_data_bad.shape[0] <= train_data_bad.shape[1]:
			return

		# more expensive cross-validation method
		# bw_estimation = 'cv_ls'

		# quick rule of thumb
		bw_estimation = 'normal_reference'

		bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes, bw=bw_estimation)
		good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

		bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
		good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

		self.kde_models[budget] = {
			'good': good_kde,
			'bad': bad_kde
		}

		# update probs for the categorical parameters for later sampling
		self.logger.debug('done building a new model for budget %f based on %i/%i split\n'
		                  'Best loss for this budget:%f\n\n\n\n\n' % (budget, n_good, n_bad, np.min(train_losses)))

