import sklearn.ensemble
from das.util.decorators import check_model
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.ParameterSpace import *


class RandomForestRegressor(BaseRegressor):

    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 e_id=None,
                 **kwargs):
        super(RandomForestRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

        self.model_name = 'RandomForestRegressor'
        self.model = sklearn.ensemble.RandomForestRegressor(n_estimators=self.n_estimators,
                                                            criterion=self.criterion,
                                                            max_depth=self.max_depth,
                                                            min_samples_split=self.min_samples_split,
                                                            min_samples_leaf=self.min_samples_leaf,
                                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                            max_features=self.max_features,
                                                            max_leaf_nodes=self.max_leaf_nodes,
                                                            min_impurity_decrease=self.min_impurity_decrease,
                                                            min_impurity_split=self.min_impurity_split,
                                                            bootstrap=self.bootstrap,
                                                            oob_score=self.oob_score,
                                                            n_jobs=self.n_jobs,
                                                            random_state=self.random_state,
                                                            verbose=self.verbose,
                                                            warm_start=self.warm_start)

    @check_model
    def apply(self, X):
        return self.model.apply(X)

    @check_model
    def decision_path(self, X):
        return self.model.decision_path(X)

    @property
    def estimators_(self):
        return self.model.estimators_

    @property
    def feature_importances_(self):
        return self.feature_importances_

    @property
    def n_features_(self):
        return self.model.n_features_

    @property
    def n_outputs_(self):
        return self.model.n_outputs_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=10, max_val=500, default=100)
            criterion_space = CategorySpace(name="criterion", choice_space=["mse", "mae"],
                                            default="mse")
            # TODO (fang xin): the min_samples_split and min_samples_leaf can be both int and float, need enhancement,
            # see sklearn introduction:
            # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
                                               default='auto')

            bootstrap_space = CategorySpace(name="bootstrap", choice_space=[True, False], default=True)

            parameter_space.merge([
                n_estimators_space,
                criterion_space,
                min_samples_split_space,
                min_samples_leaf_space,
                max_features_space,
                bootstrap_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)
        self.parameter_space = parameter_space
