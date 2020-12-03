import numpy as np
from das import ParameterSpace
from das.crossvalidate import parallel_cross_validate_score
from das.HypTuner.config_gen.config_space import ParamSpace2ConfigSpace
from das.performance_evaluation import eval_performance, score_to_loss
import das
import logging

logger = logging.getLogger(das.logger_name)
verbose = False


class CompositeModel(object):
	"""
	model: List, where the middle two elements are tuples.
	The first item of the tuple is the algorithm name (String).
	The second item of the tuple is the algorithm instance.

	The size of the model list is 4.
	The first element: (m1_name, m1) is a tuple, where model `m1` is a regressor.
	The second element: (m2_name, m2) is a tuple, where model `m2` is a regressor.
	The third element: combination strategy is a String object, 'c', 'o' or 'p'.
	Combination strategies consists of concat('c', concatenate with the raw input),
	origin('o', only raw input are fed into m2)
	and prediction('p', only the predictions of m1 are fed into m2).
	For example:
	[("ExtraTreesRegressor", sklearn.ExtraTreesRegressor()),
	("LinearRegressor", sklearn.LinearRegressor()),
	'c']

	Requirement: each algorithm instance must contain `fit` and `predict_proba` interface.
	"""
	def __init__(self, model=None, **kwargs):

		self.current_state = "start"
		self.reward = None
		if model is None and len(kwargs) > 0:
			self.set_params(**kwargs)
		elif model is not None and len(kwargs) == 0:
			self.model = model

	def fit(self, X, y=None):
		assert len(self.model) == 3, ("The length of self.model(List) should be 3,"
		                              " but {}({})".format(len(self.model), self.model))
		self.model[0][1].fit(X, y)

		y_train_ba = self.model[0][1].predict(X)[:, np.newaxis]

		concat_type = self.model[2]
		if concat_type == 'c':
			aug_x_train = np.hstack((X, y_train_ba))
		elif concat_type == 'p':
			aug_x_train = y_train_ba
		elif concat_type == 'o':
			aug_x_train = X
		else:
			raise NotImplementedError

		if np.isnan(aug_x_train).any():
			logger.info("aug_x_* has NaN~~")
			# replace nan to mean value of non-nan elements
			aug_x_train[np.isnan(aug_x_train)] = np.mean(aug_x_train[~np.isnan(aug_x_train)])

		self.model[1][1].fit(aug_x_train, y)

		return self

	def predict(self, X):

		y_test_ba = self.model[0][1].predict(X)[:, np.newaxis]

		concat_type = self.model[2]
		if concat_type == 'c':
			aug_X = np.hstack((X, y_test_ba))
		elif concat_type == 'p':
			aug_X = y_test_ba
		elif concat_type == 'o':
			aug_X = X
		else:
			raise NotImplementedError

		if np.isnan(aug_X).any():
			logger.info("aug_x_* has NaN~~")
			# replace nan to mean value of non-nan elements
			aug_X[np.isnan(aug_X)] = np.mean(aug_X[~np.isnan(aug_X)])

		y_hat = self.model[1][1].predict(aug_X)

		if np.isnan(y_hat).any():
			logger.info("aug_x_* has NaN~~")
			# replace nan to mean value of non-nan elements
			y_hat[np.isnan(y_hat)] = np.mean(y_hat[~np.isnan(y_hat)])

		return y_hat

	def predict_proba(self, X):

		return self.predict(X)

	def score(self, X, y, x_test, y_test, evaluation_rule, random_state=23, **kwargs):
		self.fit(X, y)
		y_hat = self.predict(x_test)
		test_score = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_hat, random_state=random_state)
		if 'return_dict' in kwargs:
			kwargs['return_dict']['test_score'] = test_score
		return test_score

	def get_params(self, deep=False):
		# print("self.model = {}".format(self.model))
		params_dict = {}
		for m_idx, (m_name, m_instance) in enumerate(self.model[:2]):
			params = mapping_key(m_idx, m_name, m_instance.get_params())
			params_dict.update(params)
		return params_dict

	def set_params(self, **params):
		for k in params.keys():
			m_idx, m_name, m_hyperparam = k.split("__")
			m_idx = int(m_idx)
			m_name = str(m_name)
			m_hyperparam = str(m_hyperparam)
			tmp_model = None
			for idx, m in enumerate(self.model[:2]):
				if idx == m_idx and m[0] == m_name:
					tmp_model = m[1]
					break
			if tmp_model is None:
				raise Exception("tmp_model is None, thus cannot to set params")
			if not hasattr(tmp_model.model, m_hyperparam):
				raise Exception("non-valid parameters, tmp_model.model has no hyper-param {}".format(m_hyperparam))
			setattr(tmp_model.model, m_hyperparam, params[k])
			setattr(tmp_model, m_hyperparam, params[k])

	def set_config(self, **params):
		self.set_params(**params)
		# for k in params.keys():
		# 	k_pre, _, k_post = k.partition("__")
		# 	tmp_model = None
		# 	for aa in self.model:
		# 		if aa[0] == k_pre:
		# 			tmp_model = aa[1]
		# 			break
		# 	if tmp_model is None or not hasattr(tmp_model.model, k_post):
		# 		raise Exception("non-valid parameters")
		# 	setattr(tmp_model.model, k_post, params[k])

	def get_configuration_space(self):
		tps = ParameterSpace.ParameterSpace()
		for idx, m in enumerate(self.model[:2]):
			tmp_space = m[1].get_configuration_space().get_space()
			for rr in tmp_space:
				if not rr.get_name().startswith(str(idx) + "__" + m[0] + "__"):
					rr.set_name(str(idx) + "__" + m[0] + "__" + rr.get_name())
			tps.merge(tmp_space)

		return tps

	def new_estimator(self, config=None):
		if config is not None:
			self.set_params(**config)
		return CompositeModel(model=self.model)

	def compute(self,
				config=None,
				budgets=None,
				X=None,
				y=None,
				X_val=None,
				y_val=None,
				evaluation_rule=None,
				task='regression',
	            random_state=23,
				**kwargs):
		model = self.new_estimator(config=config)
		assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"
		assert task == 'regression', "Now CompositeModel only supports regression tasks! but {}".format(task)
		# must be holdout validation strategy
		if X_val is not None:
			model.fit(X, y)
			y_hat = model.predict(X_val)
			val_score = eval_performance(rule=evaluation_rule,
										 y_true=y_val,
										 y_score=y_hat,
			                             random_state=random_state)
		# must be cross-validation strategy
		else:
			cv_fold = int(kwargs['validation_strategy_args'])
			assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
			val_score, _ = parallel_cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule,
			                                             random_state=random_state)

		# TODO(huqiu): restrict val_score to 0~1, or re-construct performance evaluation
		self.reward = {'loss': score_to_loss(evaluation_rule, val_score),
					   'info': {'val_{}'.format(evaluation_rule): val_score}}
		return self.reward

	def get_config_space(self):
		das_config_space = self.get_configuration_space()
		cs = ParamSpace2ConfigSpace(das_config_space)
		return cs

	def model_summary(self):

		if self.model is None:
			return None
		tmp = []
		for i in self.model:
			tmp.append(i[0])

		return tmp

	def get_model(self):
		return self.model

	def get_model_name(self):
		return "+".join([self.model[0][0], self.model[1][0], self.model[2]])

	def get_current_state(self):
		return self.current_state


def mapping_key(m_idx, m_name, m_params):
	new_params = {}
	for key in m_params:
		new_params["{}__{}__{}".format(m_idx, m_name, key)] = m_params[key]
	return new_params


if __name__ == "__main__":

	from das.BaseAlgorithm.Regression.SKLearnBaseAlgorithm import ExtraTreesRegressor
	from das.HyperparameterOptimizer import RandomSelector
	mm = [("ExtraTreesRegressor", ExtraTreesRegressor.ExtraTreesRegressor()),
	      ("ExtraTreesRegressor", ExtraTreesRegressor.ExtraTreesRegressor()),
	      'c']
	p = CompositeModel(mm)
	print(p.model_summary())
	cfg_space = p.get_configuration_space()
	print("cfg_space: ", cfg_space.get_space_names())
	rs = RandomSelector.RandomSelector(cfg_space)
	cand_config = rs.get_random_config()
	print(cand_config)
	p.set_params(**cand_config)


