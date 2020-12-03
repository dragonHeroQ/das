import xgboost
from das.ParameterSpace import *
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier


class XGBClassifier(BaseClassifier):

	def __init__(self,
	             max_depth=3,
	             learning_rate=0.1,
	             n_estimators=100,
	             silent=True,
	             objective='binary:logistic',
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
	             **kwargs):
		super(XGBClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
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

		self.model_name = 'XGBClassifier'
		self.model = xgboost.sklearn.XGBClassifier(max_depth=self.max_depth,
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

	def set_configuration_space(self, ps=None):
		parameter_space = ParameterSpace()
		if ps is None:
			booster_space = CategorySpace(name="booster", choice_space=['gbtree', 'gblinear', 'dart'], default='gbtree')
			learning_rate_space = LogFloatSpace(name="learning_rate", min_val=1e-2, max_val=0.5, default=0.3)
			max_depth_space = UniformIntSpace(name="max_depth", min_val=0, max_val=12, default=6)
			subsample_space = UniformFloatSpace(name="subsample", min_val=1e-2, max_val=1, default=1)
			colsample_bytree_space = UniformFloatSpace(name="colsample_bytree", min_val=1e-1, max_val=1, default=1)
			colsample_bylevel_space = UniformFloatSpace(name="colsample_bylevel", min_val=1e-1, max_val=1, default=1)
			reg_lambda_space = LogFloatSpace(name="reg_lambda", min_val=1e-10, max_val=1e-1, default=1e-10)
			reg_alpha_space = LogFloatSpace(name="reg_alpha", min_val=1e-10, max_val=1e-1, default=1e-10)
			n_estimators_space = UniformIntSpace(name="n_estimators", min_val=5, max_val=300, default=100)

			parameter_space.merge([booster_space,
			                       learning_rate_space,
			                       max_depth_space,
			                       subsample_space,
			                       colsample_bytree_space,
			                       colsample_bylevel_space,
			                       reg_lambda_space,
			                       reg_alpha_space,
			                       n_estimators_space])
		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space

		return self.parameter_space


if __name__ == "__main__":
	from sklearn.datasets import load_digits
	from sklearn.model_selection import train_test_split
	from sklearn.feature_selection import SelectKBest, f_classif

	# params = {'XGB__scale_pos_weight': 1, 'XGB__colsample_bylevel': 0.9775468894614701,
	# 'XGB__objective': 'binary:logitraw',  'XGB__reg_alpha': 6.918766708857821,
	# 'XGB__reg_lambda': 9.247559092590746, 'XGB__subsample': 0.984324605438732,
	# 'XGB__eval_metric': 'auc', 'XGB__n_estimators': 27, 'XGB__learning_rate': 0.15029925770027816,
	# 'XGB__max_depth': 4, 'XGB__colsample_tytree': 0.5090545424079975, 'XGB__booster': 'dart'}

	iris = load_digits()
	X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

	anova_filter = SelectKBest(f_classif)

	xgbclf = XGBClassifier()
	xgbclf.fit(X_train, y=Y_train)
	xgbclf.predict(X_test)
	xgbclf.predict_proba(X_test)
