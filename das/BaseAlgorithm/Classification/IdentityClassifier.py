from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator as sk_BaseEstimator
from das.ParameterSpace import *
from das.util.decorators import check_model


class FuncIdentityClassifier(sk_BaseEstimator, ClassifierMixin):
	def __init__(self):
		pass

	def fit(self, X, y, sample_weight=None):
		return X

	def predict(self, X):
		return X


class IdentityClassifier(BaseClassifier):

	def __init__(self, e_id=None, random_state=None):
		super(IdentityClassifier, self).__init__(e_id=e_id, random_state=random_state)
		self.random_state = random_state

		self.model_name = "IdentityClassifier"
		self.model = FuncIdentityClassifier()

	@check_model
	def _with_e_id_changed(self):
		pass

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		self.parameter_space = parameter_space

	def predict_proba(self, X, **predict_params):
		return X


if __name__ == "__main__":
	idenR = IdentityClassifier(random_state=0)
	X = [1, 2]
	idenR.fit(X, None)
	resX = idenR.predict(X)
	print(resX)
