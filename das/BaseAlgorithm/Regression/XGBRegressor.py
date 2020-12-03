import xgboost
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.util.decorators import check_model
from das.ParameterSpace import *


class XGBRegressor(BaseRegressor):

	def __init__(self,
	             max_depth=3,
	             learning_rate=0.1,
	             n_estimators=100,
	             silent=True,
	             objective="reg:linear",
	             booster='gbtree',
	             n_jobs=-1,
	             nthread=None,
	             gamma=0,
	             min_child_weight=1,
	             max_delta_step=0,
	             subsample=1,
	             colsample_bytree=1,
	             colsample_bylevel=1,
	             reg_alpha=0,
	             reg_lambda=1,
	             scale_pos_weight=1,
	             base_score=0.5,
	             random_state=0,
	             seed=None,
	             missing=None,
	             e_id=None,
	             **kwargs
	             ):
		super(XGBRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.silent = silent
		self.objective = objective
		self.booster = booster
		self.n_jobs = n_jobs
		self.nthread = nthread
		self.gamma = gamma
		self.min_child_weight = min_child_weight
		self.max_delta_step = max_delta_step
		self.subsample = subsample
		self.colsample_bytree = colsample_bytree
		self.colsample_bylevel = colsample_bylevel
		self.reg_alpha = reg_alpha
		self.reg_lambda = reg_lambda
		self.scale_pos_weight = scale_pos_weight
		self.base_score = base_score
		self.random_state = random_state
		self.seed = seed
		self.missing = missing

		self.model_name = 'XGBRegressor'
		self.model = xgboost.sklearn.XGBRegressor(max_depth=self.max_depth,
		                                          learning_rate=self.learning_rate,
		                                          n_estimators=self.n_estimators,
		                                          silent=self.silent,
		                                          objective=self.objective,
		                                          booster=self.booster,
		                                          n_jobs=self.n_jobs,
		                                          nthread=self.nthread,
		                                          gamma=self.gamma,
		                                          min_child_weight=self.min_child_weight,
		                                          max_delta_step=self.max_delta_step,
		                                          subsample=self.subsample,
		                                          colsample_bytree=self.colsample_bytree,
		                                          colsample_bylevel=self.colsample_bylevel,
		                                          reg_alpha=self.reg_alpha,
		                                          reg_lambda=self.reg_lambda,
		                                          scale_pos_weight=self.scale_pos_weight,
		                                          base_score=self.base_score,
		                                          random_state=self.random_state,
		                                          seed=self.seed,
		                                          missing=self.missing,
		                                          **kwargs)

	@check_model
	def apply(self, X, ntree_limit=0):
		return self.model.apply(X=X, ntree_limit=ntree_limit)

	@check_model
	def evals_result(self):
		return self.model.evals_result()

	@check_model
	def get_xgb_params(self):
		return self.model.get_xgb_params()

	@check_model
	def get_booster(self):
		return self.model.get_booster()

	@property
	def feature_importances_(self):
		return self.model.feature_importances_

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		if ps is None:
			max_depth_space = UniformIntSpace(name='max_depth', min_val=1, max_val=10, default=3)
			learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
			n_estimators_space = UniformIntSpace(name='n_estimators', min_val=5, max_val=300, default=100)
			# objective_space = CategorySpace(name='objective', choice_space=['reg:linear', ], default='reg:linear')
			booster_space = CategorySpace(name='booster', choice_space=['gbtree', 'gblinear', 'dart'], default='gbtree')
			# gamma_space = UniformFloatSpace(name='gamma', min_val=0, max_val=100, default=0)
			min_child_weight_space = UniformFloatSpace(name='min_child_weight', min_val=0, max_val=20, default=1)
			subsample_space = UniformFloatSpace(name='subsample', min_val=0.01, max_val=1.0, default=1)
			colsample_bytree_space = UniformFloatSpace(name='colsample_bytree', min_val=0.1, max_val=1, default=1)
			reg_alpha_space = LogFloatSpace(name='reg_alpha', min_val=1e-10, max_val=0.1, default=1e-10)
			reg_lambda_space = LogFloatSpace(name='reg_lambda', min_val=1e-10, max_val=0.1, default=1e-10)

			parameter_space.merge([
				booster_space,
				max_depth_space,
				learning_rate_space,
				n_estimators_space,
				# objective_space,
				min_child_weight_space,
				subsample_space,
				colsample_bytree_space,
				reg_alpha_space,
				reg_lambda_space
			])

		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space
