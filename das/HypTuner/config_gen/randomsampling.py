from .base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
	"""
    Class to implement random sampling from a ConfigSpace.

    Parameters
	----------
	config_space
		ConfigSpace.ConfigurationSpace
		The configuration space to sample from. It contains the full
		specification of the Hyper-parameters with their priors
	kwargs
		see HypTuner.config_gen.base_config_generators.BaseConfigGenerator for additional arguments
    """

	def __init__(self, config_space, **kwargs):
		super().__init__(config_space=config_space, **kwargs)

	def get_config(self, budget=None):
		return self.config_space.sample_configuration().get_dictionary(), {}
