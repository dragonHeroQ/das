from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn import tree
from das.ParameterSpace import *


class DecisionTreeClassifier(BaseClassifier):
    def __init__(self,
                 criterion='gini',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False,
                 e_id=None,
                 **kwargs):
        super(DecisionTreeClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
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
        self.class_weight = class_weight
        self.presort = presort

        self.model_name = "DecisionTreeClassifier"
        self.model = tree.DecisionTreeClassifier(criterion=self.criterion,
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
                                                 class_weight=self.class_weight,
                                                 presort=self.presort)
        self.reward = None

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            parameter_space = ParameterSpace()

            criterion_space = CategorySpace(name="criterion", choice_space=["gini", "entropy"], default="gini")
            splitter_space = CategorySpace(name="splitter", choice_space=["best", "random"], default="best")
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=5, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=5, default=1)
            # max_features_space = UniformFloatSpace(name="max_features", min_val=0.5, max_val=1, default=1)
            max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", None],
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

    def apply(self, X):
        if self.model is None:
            raise Exception
        return self.model.apply(X)

    def decision_path(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_path(X)
