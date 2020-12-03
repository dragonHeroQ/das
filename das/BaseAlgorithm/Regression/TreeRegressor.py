from das.util.decorators import check_model
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from sklearn import tree
from das.ParameterSpace import *


class DecisionTreeRegressor(BaseRegressor):

    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 presort=False,
                 e_id=None,
                 **kwargs):
        super(DecisionTreeRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort

        self.model_name = "DecisionTreeRegressor"
        self.model = tree.DecisionTreeRegressor(criterion=self.criterion,
                                                splitter=self.splitter,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_features=self.max_features,
                                                random_state=self.random_state,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_impurity_decrease=self.min_impurity_decrease,
                                                min_impurity_split=self.min_impurity_split,
                                                presort=self.presort)

    @check_model
    def apply(self, X):
        return self.model.apply(X)

    @check_model
    def decision_path(self, X):
        return self.model.decision_path(X)

    @check_model
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        return self.model.fit(X, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)

    @check_model
    def predict(self, X, check_input=True):
        return self.model.predict(X, check_input=check_input)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def max_features_(self):
        return self.model.max_features_

    @property
    def n_features_(self):
        return self.model.n_features_

    @property
    def n_outputs_(self):
        return self.model.n_outputs_

    @property
    def tree_(self):
        return self.model.tree_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            criterion_space = CategorySpace(name="criterion", choice_space=["mse", "friedman_mse", "mae"],
                                            default="mse")
            # max_depth_space = UniformFloatSpace(name="max_depth", min_val=0., max_val=2., default=0.5)
            splitter_space = CategorySpace(name="splitter", choice_space=["best", "random"], default="best")
            # TODO (fang xin): the min_samples_split and min_samples_leaf can be both int and float, need enhancement,
            # see sklearn introduction:
            # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
                                               default='auto')
            presort_space = CategorySpace(name="presort", choice_space=[True, False], default=False)

            parameter_space.merge([criterion_space,
                                   splitter_space,
                                   min_samples_split_space,
                                   min_samples_leaf_space,
                                   max_features_space,
                                   presort_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
