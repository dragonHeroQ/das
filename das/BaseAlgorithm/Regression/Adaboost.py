import sklearn
from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.util.decorators import check_model


class AdaboostRegressor(BaseRegressor):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(AdaboostRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

        self.model_name = "AdaboostRegressor"
        self.model = sklearn.ensemble.AdaBoostRegressor(base_estimator=self.base_estimator,
                                                        n_estimators=self.n_estimators,
                                                        learning_rate=self.learning_rate,
                                                        loss=self.loss,
                                                        random_state=self.random_state)

    @check_model
    def staged_predict(self, X):
        return self.model.staged_predict(X)

    @check_model
    def staged_score(self, X, y, sample_weight=None):
        return self.model.staged_score(X, y, sample_weight=sample_weight)

    @property
    def estimators_(self):
        return self.model.estimators_

    @property
    def estimator_weights_(self):
        return self.model.estimator_weights_

    @property
    def estimator_errors_(self):
        return self.model.estimator_errors_

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    def set_configuration_space(self, ps=None):

        parameter_space = ParameterSpace()
        if ps is None:
            n_estimators_space = UniformIntSpace(name='n_estimators', min_val=50, max_val=500, default=50)
            learning_rate_space = LogFloatSpace(name='learning_rate', min_val=0.01, max_val=2, default=0.1)
            loss_space = CategorySpace(name='loss', choice_space=["linear", "square", "exponential"], default="linear")

            parameter_space.merge([
                n_estimators_space,
                learning_rate_space,
                loss_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
