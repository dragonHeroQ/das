from das.ParameterSpace import *
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn.ensemble import GradientBoostingClassifier


class GBDTClassifier(BaseClassifier):

    def __init__(self,
                 loss='deviance',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto',
                 e_id=None,
                 **kwargs):

        super(GBDTClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)

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
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort

        self.model_name = "GBDTClassifier"
        self.model = GradientBoostingClassifier(loss=self.loss,
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
                                                verbose=self.verbose,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                warm_start=self.warm_start,
                                                presort=self.presort,
                                                **kwargs)
        self.reward = None

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            loss_space = CategorySpace(name="loss", choice_space=['deviance', 'exponential'], default='deviance')
            learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=50, max_val=200, default=100)
            criterion_space = CategorySpace(name="criterion", choice_space=["friedman_mse", "mse", "mae"],
                                            default="friedman_mse")
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
                                               default='auto')
            subsample_space = UniformFloatSpace(name="subsample", min_val=0.9, max_val=1.0, default=1.0)
            alpha_space = UniformFloatSpace(name="alpha", min_val=0.75, max_val=0.99, default=0.9)
            parameter_space.merge([loss_space,
                                   learning_rate_space,
                                   n_estimators_space,
                                   criterion_space,
                                   min_samples_leaf_space,
                                   min_samples_split_space,
                                   max_features_space,
                                   subsample_space,
                                   alpha_space,
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

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)
