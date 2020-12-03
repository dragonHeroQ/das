from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.performance_evaluation import eval_performance, score_to_loss
from das.util.decorators import check_model, deprecated
from das.crossvalidate import parallel_cross_validate_score
from das.util.hash_utils import get_hash


class CompositeRegressor(BaseRegressor):

	def __init__(self,
	             n_classes: int=None,
	             e_id=None,
	             random_state=None):
		super(CompositeRegressor, self).__init__(e_id=e_id, random_state=random_state, n_classes=n_classes)
		self.n_components = None

	@check_model
	def _with_e_id_changed(self):
		for i in range(self.n_components):
			self.model[i][1].set_e_id(self.gen_e_id(self.e_id, i))
			self.model[i][1].random_state = self.gen_random_state(self.e_id, i)

	@staticmethod
	def gen_random_state(e_id, m_idx=0):
		return get_hash("{}-{}".format(e_id, m_idx))

	@staticmethod
	def gen_e_id(e_id, m_idx):
		return "{}#{}".format(e_id, m_idx)

	@deprecated
	def set_config(self, **params):
		self.set_params(**params)

	def compute(self,
	            config=None,
	            budgets=None,
	            X=None,
	            y=None,
	            X_val=None,
	            y_val=None,
	            evaluation_rule=None,
	            task='classification',
	            random_state=None,
	            **kwargs):
		model = self.new_estimator(config=config)
		assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"
		# if X_val is not None, there must be holdout validation strategy
		if X_val is not None:
			model.fit(X, y)
			y_hat = model.predict(X_val)
			val_score = eval_performance(rule=evaluation_rule,
			                             y_true=y_val,
			                             y_score=y_hat,
			                             random_state=random_state)
		# if X_val is None, there must be cross-validation strategy
		else:
			cv_fold = int(kwargs['validation_strategy_args'])
			assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
			val_score, _ = parallel_cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule,
			                                             random_state=random_state)

		# TODO(huqiu): restrict val_score to 0~1, or re-construct performance evaluation
		self.reward = {'loss': score_to_loss(evaluation_rule, val_score),
		               'info': {'val_{}'.format(evaluation_rule): val_score}}
		return self.reward

	@check_model
	def model_summary(self):
		return self.get_model_name()

	@check_model
	def get_model(self):
		return self.model

	def fit(self, X, y, **fit_params):
		raise NotImplementedError

	def predict(self, X, **predict_params):
		raise NotImplementedError

	def predict_proba(self, X, **predict_params):
		raise NotImplementedError

	def predict_final_proba(self, X, **predict_params):
		raise NotImplementedError

	def score(self, X, y, evaluation_rule=None, random_state=None, **kwargs):
		y_hat = self.predict(X)
		test_score = eval_performance(rule=evaluation_rule, y_true=y, y_score=y_hat, random_state=random_state)
		if 'return_dict' in kwargs:
			kwargs['return_dict']['test_score'] = test_score
		return test_score

	@check_model
	def get_params(self, deep=True, only_cfg=True):
		params_dict = {}
		for m_idx, (m_name, m_instance) in enumerate(self.model):
			params = self.mapping_key(self.e_id, m_idx, m_name, m_instance.get_params(deep=deep, only_cfg=only_cfg))
			params_dict.update(params)
		return params_dict

	@check_model
	def set_params(self, **params):
		"""
		set_params of CompositeModel are mainly implemented in ArchiBlockClassifier and ArchiLayerClassifier.
		In ArchiBlockClassifier, we should set the SAME params to all of its components.
		In ArchiLayerClassifier, we should remember, there may be many `/` in the param_key, thus we should handle
		  it specifically.
		Hence, we leave the set_params method for its sub-classes.
		:param params: (hyper)params dict
		:return:
		"""
		raise NotImplementedError
		# for k in params.keys():
		# 	e_id, m_idx, m_name, m_hyperparam = k.split("/")
		# 	m_idx = int(m_idx)
		# 	m_name = str(m_name)
		# 	m_hyperparam = str(m_hyperparam)
		# 	tmp_model = None
		# 	for idx, m in enumerate(self.model):
		# 		if e_id == self.e_id and idx == m_idx and m[0] == m_name:
		# 			tmp_model = m[1]
		# 			break
		# 	if tmp_model is None:
		# 		raise Exception("tmp_model is None, thus cannot to set params")
		# 	if not hasattr(tmp_model.model, m_hyperparam):
		# 		raise Exception("non-valid parameters, tmp_model.model has no hyper-param {}".format(m_hyperparam))
		# 	setattr(tmp_model.model, m_hyperparam, params[k])
		# 	setattr(tmp_model, m_hyperparam, params[k])

	def get_configuration_space(self):
		raise NotImplementedError

	def new_estimator(self, config=None):
		raise NotImplementedError

	@staticmethod
	def mapping_key(e_id, m_idx, m_name, m_params):
		new_params = {}
		for key in m_params:
			new_params["{}/{}/{}/{}".format(e_id, m_idx, m_name, key)] = m_params[key]
		return new_params
