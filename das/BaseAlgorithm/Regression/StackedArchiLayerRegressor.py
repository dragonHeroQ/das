import numpy as np
from das.ParameterSpace import ParameterSpace
from das.util.decorators import check_model
from das.BaseAlgorithm.Regression.BaseCompositeRegressor import CompositeRegressor


class StackedArchiLayerRegressor(CompositeRegressor):

	def __init__(self,
	             nc: int=2,
	             model: list=None,
	             n_classes=None,
	             e_id=None,
	             random_state=None,
	             **kwargs):
		super(StackedArchiLayerRegressor, self).__init__(e_id=e_id,
		                                                 random_state=random_state, n_classes=n_classes)
		self.n_components = nc
		self.model = model or [None for _ in range(self.n_components)]

		self._with_e_id_changed()

		self.fitted = False
		self.parameter_space = None
		self.model_name = None
		self.reward = None
		for key in kwargs:
			setattr(self, key, kwargs[key])

	@check_model
	def fit(self, X, y, **fit_params):
		assert len(self.model) == self.n_components, ("The length of self.model(List) should be b={},".format(
			self.n_components) + " but {}({})".format(len(self.model), self.get_model_name()))

		X_ = X
		for i in range(self.n_components):
			self.model[i][1].fit(X_, y, **fit_params)
			proba = self.model[i][1].predict_proba(X_)
			if len(proba.shape) == 1:  # if y = (n, ), we transform it to (n, 1)
				proba = proba[:, np.newaxis]
			X_ = np.hstack((X_, proba))

		self.set_classes_(1)
		self.fitted = True
		return self

	def predict(self, X, **predict_params):
		final_proba = self.predict_final_proba(X, **predict_params)
		final_prediction = np.mean(final_proba, axis=1)
		return final_prediction

	def fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", **fit_params):
		raise NotImplementedError

	@check_model
	def predict_proba(self, X, **predict_params):
		X_ = X
		proba = np.zeros((X.shape[0], 1))
		for i in range(self.n_components):
			proba = self.model[i][1].predict_proba(X_, **predict_params)
			if len(proba.shape) == 1:  # if y = (n, ), we transform it to (n, 1)
				proba = proba[:, np.newaxis]
			X_ = np.hstack((X_, proba))

		return proba

	def predict_final_proba(self, X, **predict_params):
		return self.predict_proba(X, **predict_params)

	def new_estimator(self, config=None):
		if config is not None:
			self.set_params(**config)
		return StackedArchiLayerRegressor(model=self.model)

	@check_model
	def set_params(self, **params):
		for k in params.keys():
			things = k.split("/")
			e_id, m_idx, m_name, m_hyperparam = things[0], things[1], things[2], '/'.join(things[3:])
			m_idx = int(m_idx)
			m_name = str(m_name)
			m_hyperparam = str(m_hyperparam)
			tmp_model = None
			for idx, m in enumerate(self.model):
				# print("e_id, idx, m[0] = {}, {}, {}".format(e_id, idx, m[0]))
				# print("But **params, self.e_id, m_idx, m_name = {}, {}, {}".format(self.e_id, m_idx, m_name))
				if e_id == self.e_id and idx == m_idx and m[0] == m_name:
					tmp_model = m[1]
					break
			if tmp_model is None:
				raise Exception("tmp_model is None, thus cannot to set params")
			# Now tmp_model is a BlockModel, tmp_model.model
			tmp_model.set_params(**{m_hyperparam: params[k]})
			# setattr(tmp_model.model, m_hyperparam, params[k])
			# setattr(tmp_model, m_hyperparam, params[k])

	@check_model
	def get_configuration_space(self):
		tps = ParameterSpace()
		for idx, m in enumerate(self.model):
			tmp_space = m[1].get_configuration_space().get_space()
			for rr in tmp_space:
				if not rr.get_name().startswith("{}/{}/{}/".format(self.e_id, str(idx), m[0])):
					rr.set_name("{}/{}/{}/{}".format(self.e_id, str(idx), m[0], rr.get_name()))
			tps.merge(tmp_space)
		return tps

	def get_concise_model_name(self):
		if self.model_name is None:
			self.model_name = "[ "
			for i in range(self.n_components):
				self.model_name += self.model[i][1].get_model_name(concise=True)
				if i < self.n_components - 1:
					self.model_name += ', '
			self.model_name += " ]"
		return self.model_name

	@check_model
	def get_model_name(self, concise=False):
		if concise:
			return self.get_concise_model_name()
		if self.model_name is None:
			self.model_name = "StackedArchiLayerRegressor[\n  "
			for i in range(self.n_components):
				self.model_name += self.model[i][1].get_model_name(concise=False)
				if i < self.n_components - 1:
					self.model_name += ',\n  '
			self.model_name += "\n]"
		return self.model_name


if __name__ == '__main__':
	from das.BaseAlgorithm.Regression.RandomForestRegressor import RandomForestRegressor
	from das.BaseAlgorithm.Regression.ExtraTreesRegressor import ExtraTreesRegressor
	from das.BaseAlgorithm.Regression.ArchiBlockRegressor import HorizontalBlockRegressor
	hbc = HorizontalBlockRegressor(2, RandomForestRegressor)
	hbc2 = HorizontalBlockRegressor(2, ExtraTreesRegressor)
	alc = StackedArchiLayerRegressor(2, [("hbc", hbc), ("EXT", hbc2)], e_id=0)
	print(alc.get_model_name())
	# hbc.get_configuration_space().show_space_names()
	# alc.get_configuration_space().show_space_names()
	from benchmarks.data.mg.load_mg import load_mg

	x_train, x_test, y_train, y_test = load_mg()
	alc.fit(x_train, y_train)
	ans = alc.score(x_test, y_test, evaluation_rule='r2_score')
	print(ans)
