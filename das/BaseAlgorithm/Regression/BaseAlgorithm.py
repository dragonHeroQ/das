import numpy as np
from das.BaseAlgorithm.BaseEstimator import BaseEstimator
from das.util.decorators import check_model


class BaseRegressor(BaseEstimator):
	def __init__(self, e_id=None, random_state=None, n_classes=1, **kwargs):
		super(BaseRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
		self.classes_ = n_classes

	def predict_proba(self, X, **predict_params):
		return self.predict(X, **predict_params)[:, np.newaxis]

	def predict_log_proba(self, X, **predict_params):
		return self.predict(X, **predict_params)[:, np.newaxis]

	@property
	def num_classes(self):
		return self.classes_

	@check_model
	def classes_(self):
		if hasattr(self.model, 'classes_'):
			return self.model.classes_
		return 1

	@property
	def task(self):
		return 'regression'

	def set_classes_(self, classes: int):
		self.classes_ = classes
