import das
import copy
import logging
import numpy as np
import traceback
import ConfigSpace
import ConfigSpace.util
import scipy.stats as sps
import statsmodels.api as sm
import ConfigSpace.hyperparameters
from das.ArchitectureSearch.Optimizer.BaseOptimizer import BaseOptimizer

NO_BUDGET = 42
logger = logging.getLogger(das.logger_name)


class BayesianOptimizer(BaseOptimizer):

	def __init__(self, parameter_space,
	             min_points_in_model=None,
	             top_n_percent=15,
	             num_samples=64,
	             random_fraction=1/3,
	             bandwidth_factor=3,
	             min_bandwidth=1e-3,
	             ):
		super(BayesianOptimizer, self).__init__(parameter_space=parameter_space)
		self.top_n_percent = top_n_percent
		self.config_space = self.parameter_space.to_config_space()
		self.config_transformer = self.config_space.get_config_transformer()
		self.bw_factor = bandwidth_factor
		self.min_bandwidth = min_bandwidth

		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:   # default min points in model is dim+1
			self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1
		if self.min_points_in_model < len(self.config_space.get_hyperparameters()) + 1:
			logger.warning('Invalid min_points_in_model value.'
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

		# running attrs
		self.newest_info_dict = None

	def largest_budget_with_model(self):
		if len(self.kde_models) == 0:
			return -float('inf')
		return max(self.kde_models.keys())

	def get_debug_config(self):
		return {'b1_algo': 'MultinomialNB', 'b1_num': 2, 'b2_algo': 'GaussianNB', 'b2_num': 2}

	# TODO: budget ?
	def get_next_config(self, debug=False):
		"""
		Function to sample a new configuration.

		This function is called inside Optimizer (e.g. HyperBand) to query a new configuration.

		Parameters
		----------
		debug: float
			the budget for which this configuration is scheduled

		Returns
			(config, info_dict)
			must return a valid configuration and a (possibly empty) info dict
		-------

		"""
		if debug:
			return self.get_debug_config()
		logger.debug('start sampling a new configuration.')

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
							except Exception as e:
								logger.debug(e)
								logger.warning("Truncated Normal failed for:\ndatum=%s\n"
								               "bandwidth=%s\nfor entry with value %s" % (datum, kde_good.bw, m))
								logger.warning("data in the KDE:\n%s" % kde_good.data)
						else:
							if np.random.rand() < (1 - bw):
								vector.append(int(m))
							else:
								vector.append(np.random.randint(t))
					val = minimize_me(vector)

					if not np.isfinite(val):  # val is inf
						logger.debug('sampled vector: %s has EI value %s' % (vector, val))
						logger.debug("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
						logger.debug("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
						logger.warning("l(x) = %s" % (l(vector)))
						logger.warning("g(x) = %s" % (g(vector)))

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
					logger.debug("[BEST_VECTOR_NONE] Sampling based optimization with %i samples failed"
					             " -> using random configuration" % self.num_samples)
					sample = self.config_space.sample_configuration().get_dictionary()
					info_dict['model_based_pick'] = False
				else:
					# Sampling based optimization with samples succeed
					logger.debug(
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
			except Exception as e:
				logger.debug(e)
				logger.warning("[EXCEPTION] Sampling based optimization with %i samples failed\n %s \n"
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
			logger.warning("Error (%s) converting configuration: %s -> "
			               "using random configuration!",
			               e, sample)
			sample = self.config_space.sample_configuration().get_dictionary()
		logger.debug('done sampling a new configuration.')
		self.newest_info_dict = info_dict
		# print("Sample: {}".format(sample))
		return sample

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

	def new_result(self, config, reward, other_infos=None, update_model=True):
		"""
		Registers the result of finished runs.

		Every time a run has finished, this function should be called to register it with the result logger.
		If overwritten, make sure to call this method from the base class to ensure proper logging.

		Parameters
		----------
		config: dict
			the config of this trail
		reward: dict
			the feedback of this trail
		other_infos: dict
			other information we should to consider
		update_model: boolean
			determines whether a model inside the config_generator should be updated
		Returns
		-------
		"""
		# Empty config, we should not continue
		if config is None:
			return
		# One could skip crashed results, but we decided
		# assign a +inf loss and count them as bad configurations
		loss = np.inf if reward is None else reward['loss']

		config = copy.deepcopy(config)  # to avoid accident modification
		# if we do not consider budget, let bayesian model only containing one budget (='NO_BUDGET')
		budget = other_infos['budget'] if (isinstance(other_infos, dict)
		                                   and 'budget' in other_infos) else NO_BUDGET

		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []

		# skip model building if we already have a bigger model
		if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
			return

		# We want to get a transformed representation of the configuration in the original space
		if self.config_transformer is not None:
			# print("[BOHB_CONFIG_GEN]", self.config_transformer)
			config = self.config_transformer(config, nick2ground=False)
		conf = ConfigSpace.Configuration(self.config_space, config)
		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(loss)

		# skip model building:
		# a) if not enough points are available
		if len(self.configs[budget]) < self.min_points_in_model:
			logger.debug("Only %i run(s) for budget %f available,"
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
		logger.debug('done building a new model for budget %f based on %i/%i split\n'
		             'Best loss for this budget:%f\n\n\n\n\n' % (budget, n_good, n_bad, np.min(train_losses)))

