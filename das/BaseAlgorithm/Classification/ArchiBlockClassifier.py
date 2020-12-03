import das
import ray
import logging
import numpy as np
from das.ParameterSpace import *
from das.util.decorators import check_model
from das.util.common_utils import get_regexp_dict_value
from das.BaseAlgorithm.ray_utils import ray_fit_predict, unit_fit_predict_kfold
from das.BaseAlgorithm.Classification.BaseCompositeClassifier import CompositeClassifier

logger = logging.getLogger(das.logger_name)


class HorizontalBlockClassifier(CompositeClassifier):
	def __init__(self,
	             nc=2,
	             model_class=None,
	             model_params: dict = None,
	             n_classes=None,
	             e_id=None,
	             random_state=None,
	             **kwargs):
		super(HorizontalBlockClassifier, self).__init__(e_id=e_id, random_state=random_state, n_classes=n_classes)
		self.n_components = nc
		self.model_class = model_class
		self.model_params = model_params or self.model_class().get_params(only_cfg=True)
		logger.debug("[Block] self.model_class = {}".format(self.model_class))
		logger.debug("[Block] self.model_params = {}".format(self.model_params))
		self.model = [None for _ in range(self.n_components)]

		for i in range(self.n_components):
			self.model[i] = (self.model_class.__name__,
			                 self.model_class(e_id=self.gen_e_id(self.e_id, i),
			                                  random_state=self.gen_random_state(self.e_id, i),
			                                  **self.model_params))

		self.parameter_space = None
		self.model_name = None
		self.reward = None
		for key in kwargs:
			setattr(self, key, kwargs[key])

	@check_model
	def _with_e_id_changed(self):
		for i in range(self.n_components):
			self.model[i][1].set_e_id(self.gen_e_id(self.e_id, i))
			self.model[i][1].random_state = self.gen_random_state(self.e_id, i)
			logger.debug("E_ID_CHANGED Block, self.model[{}], e_id and random_state = {}, {}".format(
				i, self.model[i][1].e_id, self.model[i][1].random_state))

	@check_model
	def fit(self, X, y, **fit_params):
		assert len(self.model) == self.n_components, ("The length of self.model(List) should be b={},".format(
			self.n_components) + " but {}({})".format(len(self.model), self.get_model_name()))

		for i in range(self.n_components):
			logger.debug("BEFORE FIT BLOCK[{}]: random_state = {}".format(
				self.model[i][1].e_id,
				get_regexp_dict_value(self.model[i][1].get_params(only_cfg=False), 'random_state')))
			self.model[i][1].fit(X, y, **fit_params)

		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)

		return self

	def predict(self, X, **predict_params):
		final_proba = self.predict_final_proba(X, **predict_params)
		final_prediction = np.argmax(final_proba, axis=1)
		return final_prediction

	def _dis_fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", **fit_params):
		assert X_follows is not None, "At this point, X_follows should not be None!"
		result_ids = [ray_fit_predict.remote(self.model[i][1], X, y, X_follows, predict_method)
		              for i in range(self.n_components)]
		# print([type(obj_id) for obj_id in result_ids])
		# print("Spread {} tasks".format(self.n_components))
		results = ray.get(result_ids)
		# print("type_results = ", type(results))
		# results = [(res of block 1 for X_follows[0], res of block 1 for X_follows[1]),
		#            (res of block 2 for X_follows[0], res of block 2 for X_follows[1])
		#            ]
		# print("X_follows = {}".format(X_follows))
		result_pred = tuple(np.hstack((results[k][i] for k in range(self.n_components)))
		                    for i in range(len(X_follows)))
		# print("type(result_pred) = {}".format(type(result_pred)))
		# print("result_pred = {}".format(result_pred))
		return result_pred

	def fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", distribute=0, **fit_params):
		if X_follows is None:
			return None
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)
		if distribute > 1:
			return self._dis_fit_predict(X, y, X_follows, predict_method, **fit_params)
		self.fit(X, y, **fit_params)
		follow_pred = tuple(getattr(self, predict_method)(X_f)
		                    if X_f is not None else None
		                    for X_f in X_follows)
		return follow_pred

	def fit_predict_kfold_ray(self, Xid, y, Xcat, cv=3, predict_method='predict_proba',
	                          task=None, random_state=None, distribute=2):
		print("In Block fit_predict_kfold_ray")
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)
		if distribute > 1:
			print("now begin to train unit InParallel ...")
			result_ids = [unit_fit_predict_kfold.remote(self.model[i][1], Xid, y, Xcat, cv, predict_method,
			                                            task, random_state, distribute)
			              for i in range(self.n_components)]
			print("ready to get UNIT results")
			results = ray.get(result_ids)
		else:
			print("now begin to train unit sequentially ...")
			results = []
			if isinstance(Xid, ray.ObjectID):
				Xid = ray.get(Xid)
			if isinstance(Xcat, ray.ObjectID):
				Xcat = ray.get(Xcat)
			for i in range(self.n_components):
				# print("self.model[i][1] type = {}, hasattr={}".format(
				# 	type(self.model[i][1]), hasattr(self.model[i][1], 'fit_predict_kfold_ray')))
				res_unit = self.model[i][1].fit_predict_kfold_ray(Xid=Xid, y=y, Xcat=Xcat, cv=cv,
				                                                  predict_method=predict_method,
				                                                  task=task, random_state=random_state,
				                                                  distribute=distribute)
				results.append(res_unit)
			print("unit results got ...")
		for k in range(self.n_components):
			if results[k] is None:
				return None
		result_pred = np.hstack((results[k] for k in range(self.n_components)))
		print("Block hstack done")
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

	@check_model
	def get_params(self, deep=True, only_cfg=True):
		params_dict = {}
		for m_idx, (m_name, m_instance) in enumerate(self.model[:1]):
			# print("ArchiBlockClassifier m_instance({}).getparams = {}".format(type(m_instance),
			# m_instance.get_params(deep=deep, only_cfg=only_cfg)))
			params = self.mapping_key(self.e_id, 'N', m_name, m_instance.get_params(deep=deep, only_cfg=only_cfg))
			params_dict.update(params)
		# print("ArchiBlockClassifier: param_dict = {}".format(params_dict))
		return params_dict

	@check_model
	def set_params(self, **params):
		for k in params.keys():
			things = k.split("/")
			e_id, _, m_name, m_hyperparam = things[0], things[1], things[2], '/'.join(things[3:])
			# e_id, _, m_name, m_hyperparam = k.split("/")
			m_name = str(m_name)
			m_hyperparam = str(m_hyperparam)
			tmp_model = None
			for idx, m in enumerate(self.model):
				if str(e_id) == str(self.e_id) and str(m[0]) == m_name:
					tmp_model = m[1]
					at_least_one_have_this_param = False
					if hasattr(tmp_model, m_hyperparam):
						at_least_one_have_this_param = True
						setattr(tmp_model, m_hyperparam, params[k])
					if hasattr(tmp_model.model, m_hyperparam):
						at_least_one_have_this_param = True
						setattr(tmp_model.model, m_hyperparam, params[k])

					if not at_least_one_have_this_param:
						if not hasattr(tmp_model.model, m_hyperparam):
							logger.fatal(
								"tmp_model.model = {}, m_hyperparam = {}".format(type(tmp_model.model), m_hyperparam))
							logger.fatal(
								"non-valid parameters, tmp_model.model has no hyper-param {}".format(m_hyperparam))
						else:
							logger.fatal("tmp_model = {}, m_hyperparam = {}".format(type(tmp_model), m_hyperparam))
							logger.fatal("non-valid parameters, tmp_model has no hyper-param {}".format(m_hyperparam))

			if tmp_model is None:
				raise Exception("tmp_model is None, thus cannot to set params")

	@check_model
	def get_configuration_space(self):
		tps = ParameterSpace()
		for idx, m in enumerate(self.model[:1]):
			tmp_space = m[1].get_configuration_space().get_space()
			for rr in tmp_space:
				if not rr.get_name().startswith("{}/{}/{}/".format(self.e_id, 'N', m[0])):
					rr.set_name("{}/{}/{}/{}".format(self.e_id, 'N', m[0], rr.get_name()))
			tps.merge(tmp_space)
		return tps

	def new_estimator(self, config=None):
		if config is not None:
			self.set_params(**config)
		return HorizontalBlockClassifier(model=self.model)

	@check_model
	def get_model_name(self, concise=False):
		if self.model_name is None:
			self.model_name = "{}({})x{}".format(self.model_class.__name__,
			                                     self.model[0][1].get_params(only_cfg=False),
			                                     self.n_components)
		if concise:
			return "{} x {}".format(self.model_class.__name__, self.n_components)
		return self.model_name


class VerticalBlockClassifier(HorizontalBlockClassifier):

	@check_model
	def fit(self, X, y, **fit_params):
		assert len(self.model) == self.n_components, (
				"The length of self.model(List) should be b={},".format(self.n_components) +
				" but {}({})".format(len(self.model), self.get_model_name()))
		n_X = X.shape[0]
		X_ = np.zeros((n_X, 0), dtype=X.dtype)
		for i in range(self.n_components):
			X = np.hstack((X, X_))
			logger.debug("BEFORE FIT BLOCK[{}][{}]: random_state = {}".format(
				self.e_id, self.model[i][1].e_id, get_regexp_dict_value(
					self.model[i][1].get_params(only_cfg=False), 'random_state')))
			X_ = self.model[i][1].fit_predict(X, y, **fit_params)

		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)

		return self

	def fit_predict(self, X, y, X_follows=None, predict_method="predict_proba", distribute=0, **fit_params):
		if X_follows is None:
			return None
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)
		if isinstance(X, ray.ObjectID):  # For VerticalBlock, cannot distribute, so get X from plasma store
			X, y, X_follows = ray.get(X), ray.get(y), ray.get(X_follows)
		self.fit(X, y, **fit_params)
		follow_pred = tuple(getattr(self, predict_method)(X_f)
		                    if X_f is not None else None
		                    for X_f in X_follows)
		return follow_pred

	def fit_predict_kfold_ray(self, Xid, y, Xcat, cv=3, predict_method='predict_proba',
	                          task=None, random_state=None, distribute=2):
		if self.n_classes_ is None:
			num_classes = len(np.unique(y))
			self.set_classes_(num_classes)

		result_pred = np.zeros((y.shape[0], 0))
		X = np.hstack((Xid, Xcat))
		for i in range(self.n_components):
			result_pred = self.model[i][1].fit_predict_kfold_ray(Xid=X, y=y, Xcat=result_pred, cv=cv,
			                                                     predict_method=predict_method,
			                                                     task=task, random_state=random_state,
			                                                     distribute=1)

		return result_pred

	@check_model
	def predict_proba(self, X, **predict_params):
		n_X = X.shape[0]
		proba = np.zeros((n_X, 0))
		for i in range(self.n_components):
			X = np.hstack((X, proba))
			proba = self.model[i][1].predict_proba(X, **predict_params)
		return proba

	def predict_final_proba(self, X, **predict_params):
		return self.predict_proba(X, **predict_params)

	def new_estimator(self, config=None):
		if config is not None:
			self.set_params(**config)
		return VerticalBlockClassifier(model=self.model)

	@check_model
	def get_model_name(self, concise=False):
		if self.model_name is None:
			self.model_name = "X"
			for _ in range(self.n_components):
				self.model_name += "->{}({})".format(self.model_class.__name__,
				                                     self.model[0][1].get_params(only_cfg=False))
		if concise:
			self.model_name = "X"
			for _ in range(self.n_components):
				self.model_name += "->{}".format(self.model_class.__name__)
		return self.model_name


if __name__ == '__main__':
	from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
	from das.BaseAlgorithm.Classification.GBDT import GBDTClassifier

	hbc = VerticalBlockClassifier(2, GBDTClassifier, model_params={'criterion': 'mse', 'max_leaf_nodes': None,
	                                                               'max_features': 1.0, 'min_impurity_split': None,
	                                                               'min_samples_leaf': 15, 'warm_start': False,
	                                                               'validation_fraction': 0.1,
	                                                               'loss': 'deviance',
	                                                               'min_impurity_decrease': 0.0, 'max_depth': 3,
	                                                               'subsample': 1.0, 'presort': 'auto',
	                                                               'n_iter_no_change': None,
	                                                               'tol': 0.0001,
	                                                               'verbose': 0,
	                                                               # 'random_state': None,
	                                                               'init': None,
	                                                               'min_samples_split': 19,
	                                                               'min_weight_fraction_leaf': 0.0,
	                                                               'learning_rate': 0.3598678303731812,
	                                                               'n_estimators': 170})

	# hbc.get_configuration_space().show_space_names()
	print(hbc.get_model_name(concise=True))
	from benchmarks.data.letter.load_letter import load_letter
	from benchmarks.data.adult.load_adult import load_adult

	x_train, x_test, y_train, y_test = load_adult()
	hbc.fit(x_train, y_train)
	ans = hbc.score(x_test, y_test, evaluation_rule='accuracy_score')
	print(ans)
	print(hbc.get_configuration_space().show_space_names())
