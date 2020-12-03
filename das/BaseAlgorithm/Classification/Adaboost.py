from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn import ensemble
from das.ParameterSpace import *


class AdaBoostClassifier(BaseClassifier):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME',
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(AdaBoostClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        self.model_name = "AdaboostClassifier"
        self.model = ensemble.AdaBoostClassifier(base_estimator=self.base_estimator,
                                                 n_estimators=self.n_estimators,
                                                 learning_rate=self.learning_rate,
                                                 algorithm=self.algorithm,
                                                 random_state=self.random_state)

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()
            n_estimator_space = UniformIntSpace(name="n_estimators", min_val=50, max_val=300, default=100)
            learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
            # algorithm_space = CategorySpace(name="algorithm", choice_space=['SAMME'], default='SAMME')
            parameter_space.merge([    # base_estimator_space,
                n_estimator_space,
                learning_rate_space,
                # algorithm_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


if __name__ == "__main__":
    a = AdaBoostClassifier()
    a.get_configuration_space()

