import das
import ray
import sys
import logging
import numpy as np
from das.BaseAlgorithm.Classification.BaseCompositeClassifier import CompositeClassifier
from das.ParameterSpace import ParameterSpace
from das.util.decorators import check_model
from das.BaseAlgorithm.ray_utils import block_fit_predict, block_fit_predict_kfold, unit_fit_predict_kfold

logger = logging.getLogger(das.logger_name)


class ArchiLayerClassifier(CompositeClassifier):

	def __init__(self,
	             nc: int=2,
	             model: list=None,
	             n_classes=None,
	             e_id=None,
	             random_state=None,
	             **kwargs):
		super(ArchiLayerClassifier, self).__init__(e_id=e_id, random_state=random_state, n_classes=n_classes)
		self.n_components = nc
		self.model = model or [None for _ in range(self.n_components)]

		self._with_e_id_changed()

		self.fit_flag = False
		self.parameter_space = None
		self.model_name = None
		self.reward = None
		for key in kwargs:
			setattr(self, key, kwargs[key])

	@check_model
	def fit(self, X, y, **fit_params):
		assert len(self.model) == self.n_components, ("The length of self.model(List) should be b={},".format(
			self.n_components) + " but {}({})".format(len(self.model), self.get_model_name()))

		for i in range(self.n_components):
			self.model[i][1].fit(X, y, **fit_params)

		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)

		return self

	def predict(self, X, **predict_params):
		final_proba = self.predict_final_proba(X, **predict_params)
		final_prediction = np.argmax(final_proba, axis=1)
		return final_prediction

	def _dis_fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", distribute=0, **fit_params):
		assert X_follows is not None, "At this point, X_follows should not be None!"
		if isinstance(X, ray.ObjectID):
			X_id, y_id, X_follows_ids = X, y, X_follows
		else:
			X_id = ray.put(X)
			y_id = ray.put(y)
			X_follows_ids = ray.put(X_follows)
		logger.debug("Put Down! X_id({}), y_id({}) and X_follows_id({})".format(X_id, y_id, X_follows_ids))
		result_ids = [block_fit_predict.remote(self.model[i][1], X_id, y_id,
		                                       X_follows_ids, predict_method, distribute)
		              for i in range(self.n_components)]
		results = ray.get(result_ids)
		# results = [(res of block 1 for X_follows[0], res of block 1 for X_follows[1]),
		#            (res of block 2 for X_follows[0], res of block 2 for X_follows[1])
		#            ]
		if isinstance(X_follows, ray.ObjectID):
			X_follows = ray.get(X_follows)
		result_pred = tuple(np.hstack((results[k][i] for k in range(self.n_components)))
		                    for i in range(len(X_follows)))
		return result_pred

	def fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", distribute=0, **fit_params):
		if X_follows is None:
			return None
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)
		if distribute > 0:
			return self._dis_fit_predict(X, y, X_follows, predict_method, distribute, **fit_params)
		self.fit(X, y, **fit_params)
		follow_pred = tuple(getattr(self, predict_method)(X_f)
		                    if X_f is not None else None
		                    for X_f in X_follows)
		return follow_pred

	def fit_predict_kfold_ray(self, Xid, y, Xcat, cv=3, predict_method='predict_proba',
	                          task=None, random_state=None, distribute=1, **kwargs):
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)
		if distribute < 1:
			results = []
			for i in range(self.n_components):
				res_block = self.model[i][1].fit_predict_kfold_ray(Xid=Xid, y=y, Xcat=Xcat, cv=cv,
				                                                   predict_method=predict_method,
				                                                   task=task, random_state=random_state,
				                                                   distribute=distribute)
				results.append(res_block)
		elif distribute < 3:
			logger.info("ready to train block distributely ...")
			result_ids = [block_fit_predict_kfold.remote(self.model[i][1], Xid, y, Xcat, cv,
			                                             predict_method, task, random_state, distribute)
			              for i in range(self.n_components)]
			logger.info("ready to get_result_ids ...")
			results = ray.get(result_ids)
		else:  # distribute = 3, flatten blocks and units
			unit_pools = []
			for block_idx in range(self.n_components):
				block = self.model[block_idx][1]
				for unit_idx in range(block.n_components):
					unit = block.model[unit_idx][1]
					unit_pools.append(unit)
			result_ids = [unit_fit_predict_kfold.remote(unit, Xid, y, Xcat, cv,
			                                            predict_method, task, random_state, distribute)
			              for unit in unit_pools]
			logger.info("[distribute=3] ready to get unit result_ids ...")
			results = ray.get(result_ids)

		# results = [res of block 1,
		#            res of block 2]
		for k in range(len(results)):
			if results[k] is None:
				return None
		result_pred = np.hstack((results[k] for k in range(len(results))))
		return result_pred

	@check_model
	def predict_proba(self, X, **predict_params):
		output = [None for _ in range(self.n_components)]
		for i in range(self.n_components):
			output[i] = self.model[i][1].predict_proba(X, **predict_params)
		cat_output = np.hstack((output[k] for k in range(self.n_components)))
		return cat_output

	def predict_final_proba(self, X, **predict_params):
		cat_output = self.predict_proba(X, **predict_params)
		classes = self.num_classes
		final_proba = np.zeros((cat_output.shape[0], classes), dtype=X.dtype)
		for i in range(self.n_components):
			final_proba += cat_output[:, i * classes:(i + 1) * classes]
		final_proba /= self.n_components
		return final_proba

	def new_estimator(self, config=None):
		if config is not None:
			self.set_params(**config)
		return ArchiLayerClassifier(model=self.model)

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
			self.model_name = "[\n  "
			for i in range(self.n_components):
				self.model_name += self.model[i][1].get_model_name(concise=False)
				if i < self.n_components - 1:
					self.model_name += ',\n  '
			self.model_name += "\n]"
		return self.model_name


if __name__ == '__main__':
	from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
	from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier
	hbc = HorizontalBlockClassifier(2, RandomForestClassifier)
	hbc2 = HorizontalBlockClassifier(2, RandomForestClassifier)
	alc = ArchiLayerClassifier(2, [("hbc", hbc), ("EXT", hbc2)], e_id=0)
	print(alc.get_model_name())
	hbc.get_configuration_space().show_space_names()
	alc.get_configuration_space().show_space_names()
	# from benchmarks.letter.load_letter import load_letter
	#
	# x_train, x_test, y_train, y_test = load_letter()
	# alc.fit(x_train, y_train)
	# ans = alc.score(x_test, y_test, evaluation_rule='accuracy_score')
	# print(ans)
