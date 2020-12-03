from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor as sk_ExtraTreesRegressor


class ExtraTreesRegressor(BaseRegressor):

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
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 e_id=None):
        super(ExtraTreesRegressor, self).__init__(e_id=e_id, random_state=random_state)
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

        self.model_name = "ExtraTreesRegressor"
        self.model = sk_ExtraTreesRegressor(n_estimators=self.n_estimators,
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

    @property
    def estimators_(self):
        return self.model.estimators_

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def n_features_(self):
        return self.model.n_features_

    @property
    def n_outputs_(self):
        return self.model.n_outputs_

    @property
    def oob_score_(self):
        return self.model.oob_score_

    @property
    def oob_prediction_(self):
        return self.model.oob_prediction_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=10, max_val=500, default=100)
            criterion_space = CategorySpace(name="criterion", choice_space=["mse", "mae"], default="mse")
            max_features_space = UniformFloatSpace(name="max_features", min_val=0.1, max_val=1.0, default=1)
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            bootstrap_space = CategorySpace(name="bootstrap", choice_space=[True, False], default=False)

            parameter_space.merge([
                n_estimators_space,
                criterion_space,
                max_features_space,
                min_samples_split_space,
                min_samples_leaf_space,
                bootstrap_space
            ])

        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def apply(self, X):
        if self.model is None:
            raise Exception
        return self.model.apply(X)

    def decision_path(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_path(X)
