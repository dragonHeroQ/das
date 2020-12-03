from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator as sk_BaseEstimator
from das.ParameterSpace import *
from das.util.decorators import check_model


class FuncIdentityRegressor(sk_BaseEstimator, RegressorMixin):
	def __init__(self):
		pass

	def fit(self, X, y, sample_weight=None):
		return X

	def predict(self, X):
		return X


class IdentityRegressor(BaseRegressor):

	def __init__(self, e_id=None, random_state=None):
		super(IdentityRegressor, self).__init__(e_id=e_id, random_state=random_state)
		self.random_state = random_state

		self.model_name = "IdentityRegressor"
		self.model = FuncIdentityRegressor()

	@check_model
	def _with_e_id_changed(self):
		pass

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		self.parameter_space = parameter_space

	def predict_proba(self, X, **predict_params):
		return X


if __name__ == "__main__":
	idenR = IdentityRegressor(random_state=0)
	X = [1, 2]
	idenR.fit(X, None)
	resX = idenR.predict(X)
	print(resX)
