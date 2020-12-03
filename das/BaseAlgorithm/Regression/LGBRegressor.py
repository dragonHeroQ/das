import lightgbm as lgb
from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.util.decorators import check_model


class LGBRegressor(BaseRegressor):

	def __init__(self,
	             boosting_type="gbdt",
	             num_leaves=31,
	             max_depth=-1,
	             learning_rate=0.1,
	             n_estimators=100,
	             subsample_for_bin=200000,
	             objective='regression',
	             class_weight=None,
	             min_split_gain=0.,
	             min_child_weight=1e-3,
	             min_child_samples=20,
	             subsample=1.,
	             subsample_freq=0,
	             colsample_bytree=1.,
	             reg_alpha=0.,
	             reg_lambda=0.,
	             random_state=None,
	             n_jobs=-1,
	             silent=True,
	             importance_type='split',
	             e_id=None,
	             **kwargs
	             ):
		super(LGBRegressor, self).__init__(e_id=e_id, random_state=random_state)

		self.boosting_type = boosting_type
		self.num_leaves = num_leaves
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.subsample_for_bin = subsample_for_bin
		self.objective = objective
		self.class_weight = class_weight
		self.min_split_gain = min_split_gain
		self.min_child_weight = min_child_weight
		self.min_child_samples = min_child_samples
		self.subsample = subsample
		self.subsample_freq = subsample_freq
		self.colsample_bytree = colsample_bytree
		self.reg_alpha = reg_alpha
		self.reg_lambda = reg_lambda
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.silent = silent
		self.importance_type = importance_type
		self.kwargs = kwargs

		self.model_name = 'LGBRegressor'
		self.model = lgb.sklearn.LGBMRegressor(boosting_type=self.boosting_type,
		                                       num_leaves=self.num_leaves,
		                                       max_depth=self.max_depth,
		                                       learning_rate=self.learning_rate,
		                                       n_estimators=self.n_estimators,
		                                       subsample_for_bin=self.subsample_for_bin,
		                                       objective=self.objective,
		                                       class_weight=self.class_weight,
		                                       min_split_gain=self.min_split_gain,
		                                       min_child_weight=self.min_child_weight,
		                                       min_child_samples=self.min_child_samples,
		                                       subsample=self.subsample,
		                                       subsample_freq=self.subsample_freq,
		                                       colsample_bytree=self.colsample_bytree,
		                                       reg_alpha=self.reg_alpha,
		                                       reg_lambda=self.reg_lambda,
		                                       random_state=self.random_state,
		                                       n_jobs=self.n_jobs,
		                                       silent=self.silent,
		                                       importance_type=self.importance_type,
		                                       **self.kwargs)

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		if ps is None:
			max_depth_space = UniformIntSpace(name='max_depth', min_val=1, max_val=10, default=-1)
			learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
			n_estimators_space = UniformIntSpace(name='n_estimators', min_val=5, max_val=300, default=100)
			min_child_weight_space = UniformFloatSpace(name='min_child_weight', min_val=0, max_val=20, default=1)
			subsample_space = UniformFloatSpace(name='subsample', min_val=0.01, max_val=1.0, default=1)
			colsample_bytree_space = UniformFloatSpace(name='colsample_bytree', min_val=0.1, max_val=1, default=1)
			reg_alpha_space = LogFloatSpace(name='reg_alpha', min_val=1e-10, max_val=0.1, default=1e-10)
			reg_lambda_space = LogFloatSpace(name='reg_lambda', min_val=1e-10, max_val=0.1, default=1e-10)

			parameter_space.merge([
				# max_depth_space,
				learning_rate_space,
				n_estimators_space,
				# booster_space,
				# gamma_space,
				# min_child_weight_space,
				subsample_space,
				colsample_bytree_space,
				# reg_alpha_space,
				# reg_lambda_space
			])

		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space
