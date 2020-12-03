import numpy as np


class HalvingStrategy(object):
	pass


class SuccessiveHalving(HalvingStrategy):
	"""
	SuccessiveHalving Strategy. see HyperBand paper.
	"""
	def _advance_to_next_stage(self, config_ids, losses):
		"""
		SuccessiveHalving simply continues the best based on the current loss.
		"""
		ranks = np.argsort(np.argsort(losses))
		return ranks < self.num_configs[self.stage]



