from das.util.decorators import check_model
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.ParameterSpace import *

from sklearn.ensemble import GradientBoostingRegressor


class GBDTRegressor(BaseRegressor):

    def __init__(self,
                 loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto',
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 e_id=None):

        super(GBDTRegressor, self).__init__(e_id=e_id, random_state=random_state)

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        self.model_name = "GBDTRegressor"
        self.model = GradientBoostingRegressor(loss=self.loss,
                                               learning_rate=self.learning_rate,
                                               n_estimators=self.n_estimators,
                                               subsample=self.subsample,
                                               criterion=self.criterion,
                                               min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf,
                                               min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                               max_depth=self.max_depth,
                                               min_impurity_decrease=self.min_impurity_decrease,
                                               min_impurity_split=self.min_impurity_split,
                                               init=self.init,
                                               random_state=self.random_state,
                                               max_features=self.max_features,
                                               alpha=self.alpha,
                                               verbose=self.verbose,
                                               max_leaf_nodes=self.max_leaf_nodes,
                                               warm_start=self.warm_start,
                                               presort=self.presort,
                                               validation_fraction=self.validation_fraction,
                                               n_iter_no_change=self.n_iter_no_change,
                                               tol=self.tol)

    @check_model
    def apply(self, X):
        return self.model.apply(X)

    @check_model
    def staged_predict(self, X):
        return self.model.staged_predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def oob_improvement_(self):
        return self.model.oob_improvement_

    @property
    def train_score_(self):
        return self.model.train_score_

    @property
    def loss_(self):
        return self.model.loss_

    @property
    def init_(self):
        return self.model.init_

    @property
    def estimators_(self):
        return self.model.estimators_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            loss_space = CategorySpace(name="loss", choice_space=["ls", "lad", "huber", "quantile"], default='ls')
            learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=50, max_val=500, default=100)
            criterion_space = CategorySpace(name="criterion", choice_space=["friedman_mse", "mse", "mae"],
                                            default="friedman_mse")
            max_depth_space = UniformIntSpace(name="max_depth", min_val=1, max_val=10, default=3)
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            subsample_space = UniformFloatSpace(name="subsample", min_val=0.9, max_val=1.0, default=1.0)
            max_features_space = UniformFloatSpace(name="max_features", min_val=0.1, max_val=1.0, default=1)
            alpha_space = UniformFloatSpace(name="alpha", min_val=0.75, max_val=0.99, default=0.9)

            parameter_space.merge([
                                   loss_space,
                                   learning_rate_space,
                                   n_estimators_space,
                                   criterion_space,
                                   max_depth_space,
                                   min_samples_leaf_space,
                                   min_samples_split_space,
                                   subsample_space,
                                   max_features_space,
                                   alpha_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)