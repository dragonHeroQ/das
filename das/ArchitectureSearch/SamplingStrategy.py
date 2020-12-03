import das
import logging
import numpy as np
from das.util.common_utils import getmbof

logger = logging.getLogger(das.logger_name)


class SamplingStrategy(object):

	def __init__(self):
		pass

	def sample(self, X, y, **kwargs):
		raise NotImplementedError


class SizeSamplingStrategy(SamplingStrategy):

	def __init__(self, fidelity_mb=1.0):
		super().__init__()
		self.fidelity_mb = fidelity_mb

	@staticmethod
	def sampling_fidelity(X, mb=1.0):
		low = 1
		high = X.shape[0]
		while low <= high:
			mid = (low + high) // 2
			if getmbof(X[:mid]) < mb:
				low = mid + 1
			else:
				high = mid - 1
		return high

	def sample(self, X, y, random_state=None, **kwargs):
		mb = self.fidelity_mb
		if getmbof(X) > mb:
			fidelity = self.sampling_fidelity(X=X, mb=mb)
			logger.info("Proper Fidelity: {}".format(fidelity))
			np.random.seed(random_state)
			indexes = np.random.choice(len(X), fidelity)
			x_train_fidelity = X[indexes]
			y_train_fidelity = y[indexes]
		else:
			x_train_fidelity = X
			y_train_fidelity = y
		return x_train_fidelity, y_train_fidelity

