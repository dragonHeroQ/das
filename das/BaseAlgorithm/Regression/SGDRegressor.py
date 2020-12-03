from sklearn import linear_model
from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.util.decorators import check_model

DEFAULT_EPSILON = 0.1


class SGDRegressor(BaseRegressor):

    def __init__(self,
                 loss="squared_loss",
                 penalty="l2",
                 alpha=0.0001,
                 l1_ratio=0.15,
                 fit_intercept=True,
                 max_iter=None,
                 tol=None,
                 shuffle=True,
                 verbose=0,
                 epsilon=DEFAULT_EPSILON,
                 random_state=None,
                 learning_rate="invscaling",
                 eta0=0.01,
                 power_t=0.25,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 warm_start=False,
                 average=False,
                 n_iter=None,
                 e_id=None,
                 **kwargs):
        super(SGDRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.n_iter = n_iter

        self.model_name = "SGDRegressor"
        self.model = linear_model.SGDRegressor(loss=self.loss,
                                               penalty=self.penalty,
                                               alpha=self.alpha,
                                               l1_ratio=self.l1_ratio,
                                               fit_intercept=self.fit_intercept,
                                               max_iter=self.max_iter,
                                               tol=self.tol,
                                               shuffle=self.shuffle,
                                               verbose=self.verbose,
                                               epsilon=self.epsilon,
                                               random_state=self.random_state,
                                               learning_rate=self.learning_rate,
                                               eta0=self.eta0,
                                               power_t=self.power_t,
                                               early_stopping=self.early_stopping,
                                               validation_fraction=self.validation_fraction,
                                               n_iter_no_change=self.n_iter_no_change,
                                               warm_start=self.warm_start,
                                               average=self.average,
                                               n_iter=self.n_iter)

    @check_model
    def densify(self):
        return self.model.densify()

    @check_model
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        return self.model.fit(X, y, coef_init=coef_init, intercept_init=intercept_init, sample_weight=sample_weight)

    @check_model
    def partial_fit(self, X, y, sample_weight=None):
        return self.model.partial_fit(X, y, sample_weight=sample_weight)

    @check_model
    def sparsify(self):
        return self.model.sparsify()

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def average_coef_(self):
        return self.model.average_coef_

    @property
    def average_intercept_(self):
        return self.model.average_intercept_

    @property
    def n_iter_(self):
        return self.model.n_iter_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            parameter_space = ParameterSpace()

            loss_space = CategorySpace(name="loss",
                                       choice_space=["squared_loss", "huber", "epsilon_insensitive",
                                                     "squared_epsilon_insensitive"],
                                       default="squared_loss")
            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2", "elasticnet", "none"], default="l2")
            alpha_space = LogFloatSpace(name="alpha", min_val=1e-7, max_val=1e-1, default=1e-4)
            l1_ratio_space = LogFloatSpace(name="l1_ratio", min_val=1e-9, max_val=1., default=0.15)
            tol_space = LogFloatSpace(name="tol", min_val=1e-5, max_val=1e-1, default=0.1)
            epsilon_space = LogFloatSpace(name="epsilon", min_val=1e-5, max_val=1e-1, default=0.1)
            learning_rate_space = CategorySpace(name="learning_rate",
                                                choice_space=["constant", "optimal", "invscaling", "adaptive"],
                                                default="invscaling")
            eta0_space = LogFloatSpace(name="eta0", min_val=1e-7, max_val=1e-1, default=0.01)
            power_t_space = UniformFloatSpace(name="power_t", min_val=1e-5, max_val=1, default=0.25)
            average_space = CategorySpace(name="average", choice_space=[True, False], default=False)

            parameter_space.merge([loss_space,
                                   penalty_space,
                                   alpha_space,
                                   l1_ratio_space,
                                   # max_iter_space,
                                   tol_space,
                                   epsilon_space,
                                   # shuffle_space,
                                   learning_rate_space,
                                   eta0_space,
                                   power_t_space,
                                   average_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
