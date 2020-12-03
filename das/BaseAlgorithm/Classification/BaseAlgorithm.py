from das.BaseAlgorithm.BaseEstimator import BaseEstimator


class BaseClassifier(BaseEstimator):

	def __init__(self, e_id=None, random_state=None, n_classes=None, **kwargs):
		super(BaseClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
		self.n_classes_ = n_classes

	@property
	def task(self):
		return 'classification'

	def set_classes_(self, classes: int):
		self.n_classes_ = classes

	@property
	def num_classes(self):
		return self.n_classes_
