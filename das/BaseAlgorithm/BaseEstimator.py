import das
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from das.ParameterSpace import *
from das.util.hash_utils import get_hash
from das.crossvalidate import fit_and_transform_v2
from das.crossvalidate import cross_validate_score
from sklearn.model_selection import StratifiedKFold, KFold
from das.performance_evaluation import eval_performance, score_to_loss
from das.util.decorators import check_parameter_space, check_model

logger = logging.getLogger(das.logger_name)


class BaseEstimator(object):

	def __init__(self, e_id=None, random_state=None, **kwargs):
		self.model = None
		self.parameter_space = None
		self.model_name = None
		self.reward = None
		self.random_state = random_state
		self.e_id = str(e_id)
		for key in kwargs:
			setattr(self, key, kwargs)

	@check_model
	def fit(self, X, y, **fit_params):
		true_fit_params = {}
		for key in fit_params:
			if key in ['sample_weight']:
				true_fit_params[key] = fit_params[key]
		self.model.fit(X, y, **true_fit_params)
		return self

	@check_model
	def fit_predict(self, X, y, X_follows=None, predict_method='predict_proba', **fit_params):
		if X_follows is None:
			return None
		true_fit_params = {}
		for key in fit_params:
			if key in ['sample_weight']:
				true_fit_params[key] = fit_params[key]
		self.fit(X, y, **true_fit_params)
		if not isinstance(X_follows, (tuple, list)):
			X_follows = [X_follows]
		return [getattr(self, predict_method)(X_follow) for X_follow in X_follows]

	def fit_predict_kfold_ray(self, Xid, y, Xcat, cv, predict_method='predict_proba',
	                          task=None, random_state=None, distribute=1):
		print("In BaseEstimator")
		X = np.hstack((Xid, Xcat))
		row_n = y.shape[0]
		y_final_val_pred = None
		if (len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)) and task != "regression":
			# logger.debug("start cv, StratifiedKFold")
			splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
		else:
			# logger.debug("start cv, KFold")
			splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

		print("Unit FitPredict KFold: Concatenated X shape = {}".format(X.shape))
		train_val_idx = list(splitter.split(X, y))

		predictions = []
		for train_idx, val_idx in train_val_idx:
			predictions.append(fit_and_transform_v2(
				model=self, X=X, y=y, train_idx=train_idx, val_idx=val_idx,
				X_follow=None, predict_method=predict_method, distribute=distribute))
		print("KFold done")
		for i, (y_val_pred, _) in enumerate(predictions):
			if y_val_pred is None:  # if any exception caused None for y_val_pred, we directly return None
				return None
			if i == 0:
				shape_1 = y_val_pred.shape[1]
				y_final_val_pred = np.zeros((row_n, shape_1))

			y_final_val_pred[train_val_idx[i][1]] = y_val_pred

		return y_final_val_pred

	@check_model
	def predict(self, X, **predict_params):
		if X is None:
			return None
		# now predict do not accept arguments unless X
		return self.model.predict(X)

	@check_model
	def predict_proba(self, X, **predict_params):
		if X is None:
			return None
		# now predict_proba do not accept arguments unless X
		return self.model.predict_proba(X)

	@check_model
	def predict_log_proba(self, X, **predict_params):
		assert hasattr(self.model, 'predict_log_proba'), ("Model {} has not attr:"
		                                                  " predict_log_proba".format(self.get_model_name()))
		if X is None:
			return None
		# now predict_log_proba do not accept arguments unless X
		return self.model.predict_log_proba(X)

	@check_model
	def score(self, X, y, **score_params):
		return self.model.score(X, y, **score_params)

	@staticmethod
	def roc_auc_score(y_true, y_score, average=None, sample_weight=None):
		return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight)

	@check_model
	def get_params(self, deep=True, only_cfg=True):
		"""
		:param deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.
		:param only_cfg: boolean, optional
			If True, will return the parameters for this estimator that
			only included in the configuration space.
		:return:
		"""
		param_dict = self.model.get_params(deep=deep)
		if not only_cfg:
			return param_dict
		new_param_dict = {}
		space_names = self.get_configuration_space().get_space_names()
		for key in param_dict.keys():
			yes = False
			for space_name in space_names:
				if key in space_name:
					yes = True
					break
			if yes:
				new_param_dict[key] = param_dict[key]

		return new_param_dict

	@check_model
	def get_model_type(self):
		return self.model._estimator_type

	@check_model
	def set_config(self, params):
		for k, v in params.items():
			if not hasattr(self.model, k):
				raise TypeError("There is no attribute named %s, %s" % (k, self.model))

			setattr(self, k, v)

	@check_model
	def set_params(self, **kwargs):
		return self.model.set_params(**kwargs)

	def gen_random_state(self, e_id, m_idx=0):
		return get_hash("{}_{}".format(e_id, self.random_state))

	@check_model
	def _with_e_id_changed(self):
		if 'random_state' in self.model.get_params():
			self.model.set_params(random_state=self.gen_random_state(self.e_id))

	@check_model
	def set_e_id(self, e_id=None):
		self.e_id = e_id
		self._with_e_id_changed()

	@check_parameter_space
	def get_configuration_space(self):
		if self.parameter_space is None:
			self.set_configuration_space()
		return self.parameter_space

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		self.parameter_space = parameter_space

	@check_model
	def new_estimator(self, config):
		self.model.set_params(**config)

	def compute(self, config_id=None, config=None, budgets=None, X=None, y=None,
	            X_val=None, y_val=None, evaluation_rule=None, task='classification',
	            working_directory=".", *args, **kwargs):
		model = self.new_estimator(config=config)
		assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"
		if task == 'clustering':
			model.fit(X)
			print("X.shape=", X.shape)
			y_hat = model.predict(X)
			print("y_hat.shape", y_hat.shape)
			val_score = eval_performance(rule=evaluation_rule,
			                             X=X,
			                             y_score=y_hat)
		elif X_val is not None:
			model.fit(X, y)
			y_hat = model.predict(X_val)
			val_score = eval_performance(rule=evaluation_rule,
			                             y_true=y_val,
			                             y_score=y_hat)
		else:
			cv_fold = kwargs['validation_strategy_args']
			assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
			val_score, _ = cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule)
		self.reward = {'loss': score_to_loss(evaluation_rule, val_score),
		               'info': {'val_{}'.format(evaluation_rule): val_score}}
		return self.reward

	def get_config_space(self):
		das_config_space = self.get_configuration_space()
		cs = ParamSpace2ConfigSpace(das_config_space)
		return cs

	def get_model_name(self):
		return self.model_name

	@check_model
	def get_model(self):
		return self.model

	def __repr__(self):
		if self.model_name is not None:
			return str(self.model_name)
		return str(self.__class__.__name__)
