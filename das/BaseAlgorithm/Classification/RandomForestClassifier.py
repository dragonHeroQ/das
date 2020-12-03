from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn import ensemble
from das.ParameterSpace import *
from das.util.ParameterRelation import *


class RandomForestClassifier(BaseClassifier):

    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 e_id=None,
                 **kwargs):
        super(RandomForestClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)

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
        self.class_weight = class_weight

        self.model_name = "RandomForestClassifier"
        self.model = ensemble.RandomForestClassifier(n_estimators=self.n_estimators,
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
                                                     warm_start=self.warm_start,
                                                     class_weight=self.class_weight)
        self.reward = None

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=10, max_val=500, default=100)
            criterion_space = CategorySpace(name="criterion", choice_space=["gini", "entropy"], default="gini")
            max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
                                               default='auto')

            # max_depth_space
            max_depth_space = UniformIntSpace(name='max_depth', min_val=2, max_val=100, default=100)
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=20, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=20, default=1)
            # Out of bag estimation only available if bootstrap=True
            bootstrap_space = CategorySpace(name="bootstrap", choice_space=[True], default=True)
            oob_score_space = CategorySpace(name="oob_score", choice_space=[True, False], default=False)

            # bootstrap_oobscore_relation = ConditionRelation((oob_score_space.get_name, True),
            #                                                 (bootstrap_space.get_name, True))

            # self.parameter_space.add_parameter_relation(bootstrap_oobscore_relation)

            parameter_space.merge([n_estimators_space,
                                   criterion_space,
                                   max_depth_space,
                                   max_features_space,
                                   min_samples_split_space,
                                   min_samples_leaf_space,
                                   bootstrap_space,
                                   oob_score_space])
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
