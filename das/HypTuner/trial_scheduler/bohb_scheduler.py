import numpy as np
from das.HypTuner.iteration.iteration_strategy import SuccessiveHalving
from das.HypTuner.trial_scheduler.trial_scheduler import TrialScheduler
from das.HypTuner.config_gen.bohb_config_gen import BOHBConfigGenerator


class BOHBScheduler(TrialScheduler):
	"""
	BOHB performs robust and efficient HyperParameter optimization
	at scale by combining the speed of HyperBand searches with the
	guidance and guarantees of convergence of Bayesian
	Optimization. Instead of sampling new configurations at random,
	BOHB uses kernel density estimators to select promising candidates.

	.. highlight:: none

	For reference: ::

		@InProceedings{falkner-icml-18,
		  title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
		  author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
		  booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
		  pages =        {1436--1445},
		  year =         {2018},
		}

	Parameters
	----------
	config_space: ConfigSpace object
		valid representation of the search space
	eta : float
		In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
		Must be greater or equal to 2.
	min_budget : float
		The smallest budget to consider. Needs to be positive!
	max_budget : float
		The largest budget to consider. Needs to be larger than min_budget!
		The budgets will be geometrically distributed
					:math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
	min_points_in_model: int
		number of observations to start building a KDE. Default 'None' means
		dim+1, the bare minimum.
	top_n_percent: int
		percentage ( between 1 and 99, default 15) of the observations that are considered good.
	num_samples: int
		number of samples to optimize EI (default 64)
	random_fraction: float
		fraction of purely random configurations that are sampled from the
		prior without the model.
	bandwidth_factor: float
		to encourage diversity, the points proposed to optimize EI, are sampled
		from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3)
	min_bandwidth: float
		to keep diversity, even when all (good) samples have the same value for one of the parameters,
		a minimum bandwidth (Default: 1e-3) is used instead of zero.
	iteration_kwargs: dict
		kwargs to be added to the instantiation of each iteration
	"""

	def __init__(self,
	             config_space=None,
	             eta=3,
	             min_budget=3,
	             max_budget=81,
	             min_points_in_model=None,
	             top_n_percent=15,
	             num_samples=64,
	             random_fraction=1 / 3,
	             bandwidth_factor=3,
	             min_bandwidth=1e-3,
	             **kwargs):
		# TODO: Proper check for ConfigSpace object!
		if config_space is None:
			raise ValueError("You have to provide a valid ConfigSpace object")
		self.config_space = config_space

		config_gentor = BOHBConfigGenerator(config_space=config_space,
		                                    min_points_in_model=min_points_in_model,
		                                    top_n_percent=top_n_percent,
		                                    num_samples=num_samples,
		                                    random_fraction=random_fraction,
		                                    bandwidth_factor=bandwidth_factor,
		                                    min_bandwidth=min_bandwidth)

		super().__init__(config_generator=config_gentor, **kwargs)

		# HyperBand related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget

		# Pre-compute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

		self.sched_config.update({
			'eta': eta,
			'min_budget': min_budget,
			'max_budget': max_budget,
			'budgets': self.budgets,
			'max_SH_iter': self.max_SH_iter,
			'min_points_in_model': min_points_in_model,
			'top_n_percent': top_n_percent,
			'num_samples': num_samples,
			'random_fraction': random_fraction,
			'bandwidth_factor': bandwidth_factor,
			'min_bandwidth': min_bandwidth
		})

	def get_next_iteration(self, iteration, iteration_kwargs=None):
		"""
		BO-HB uses (just like HyperBand) SuccessiveHalving for each iteration.
		See Li et al. (2016) for reference.

		Parameters
		----------
		iteration: int
			the index of the iteration to be instantiated
		iteration_kwargs: dict
			key word arguments for the iteration

		Returns
		-------
		SuccessiveHalving: the SuccessiveHalving iteration with the
			corresponding number of configurations
		"""

		# number of 'SH runs'
		s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
		# number of configurations in that bracket
		n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
		ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
		nr = self.budgets[(-s - 1):]
		if iteration_kwargs is None:
			iteration_kwargs = {}
		# print("This Iteration(1): self.budget = {}".format(self.budgets))
		# print("This Iteration(2): bracket ({}, {}), s = {}, ns = {}, nr = {}".format(n0, nr[0], s, ns, nr))
		return (SuccessiveHalving(cur_iter=iteration,
		                          num_configs=ns,
		                          budgets=nr,
		                          config_generator=self.config_generator,
		                          budget_type=self.budget_type,
		                          min_budget=self.min_budget,
		                          max_budget=self.max_budget,
		                          **iteration_kwargs))


if __name__ == '__main__':
	pass
