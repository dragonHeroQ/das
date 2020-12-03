from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
import sklearn
from das.ParameterSpace import *


class SVR(BaseRegressor):

    def __init__(self,
                 kernel='rbf',
                 degree=3,
                 gamma='auto_deprecated',
                 coef0=0.0,
                 tol=1e-3,
                 C=1.0,
                 epsilon=0.1,
                 shrinking=True,
                 cache_size=200,
                 verbose=False,
                 max_iter=-1,
                 e_id=None,
                 random_state=None,
                 **kwargs):

        super(SVR, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

        self.model_name = "epsilon_svr"
        self.model = sklearn.svm.SVR(kernel=self.kernel,
                                     degree=self.degree,
                                     gamma=self.gamma,
                                     coef0=self.coef0,
                                     tol=self.tol,
                                     C=self.C,
                                     epsilon=self.epsilon,
                                     shrinking=self.shrinking,
                                     cache_size=self.cache_size,
                                     verbose=self.verbose,
                                     max_iter=self.max_iter)

    @property
    def support_(self):
        return self.model.support_

    @property
    def support_vectors_(self):
        return self.model.support_vectors_

    @property
    def dual_coef_(self):
        return self.model.dual_coef_

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            kernel_space = CategorySpace(name="kernel",
                                         choice_space=["linear", "poly", "rbf", "sigmoid", "precomputed"], default='rbf')
            degree_space = UniformIntSpace(name="degree", min_val=2, max_val=10, default=3)
            gamma_space = CategorySpace(name="gamma", choice_space=["scale", "auto_deprecated"],
                                        default="auto_deprecated")
            tol_space = LogFloatSpace(name="tol", min_val=1e-4, max_val=1e-2, default=1e-3)
            C_space = LogFloatSpace(name="C", min_val=1e-2, max_val=1e2, default=1.0)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            parameter_space.merge([kernel_space,
                                   degree_space,
                                   gamma_space,
                                   tol_space,
                                   C_space,
                                   shrinking_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class NuSVR(BaseRegressor):
    def __init__(self,
                 nu=0.5,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto_deprecated',
                 coef0=0.0,
                 shrinking=True,
                 tol=1e-3,
                 cache_size=200,
                 verbose=False,
                 max_iter=-1,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(NuSVR, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.nu = nu
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

        self.model_name = "NU_SVR"
        self.model = sklearn.svm.NuSVR(nu=self.nu,
                                       C=self.C,
                                       kernel=self.kernel,
                                       degree=self.degree,
                                       gamma=self.gamma,
                                       coef0=self.coef0,
                                       shrinking=self.shrinking,
                                       tol=self.tol,
                                       cache_size=self.cache_size,
                                       verbose=self.verbose,
                                       max_iter=self.max_iter
                                       )

    @property
    def support_(self):
        return self.model.support_

    @property
    def support_vectors_(self):
        return self.model.support_vectors_

    @property
    def dual_coef_(self):
        return self.model.dual_coef_

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            # TODO(fang xin): the nu should be in the interval (0,1]
            nu_space = UniformFloatSpace(name="nu", min_val=1e-4, max_val=1-1e-4, default=0.5)
            kernel_space = CategorySpace(name="kernel",
                                         choice_space=["linear", "poly", "rbf", "sigmoid", "precomputed"], default="rbf")
            degree_space = UniformIntSpace(name="degree", min_val=2, max_val=10, default=3)
            gamma_space = CategorySpace(name="gamma", choice_space=["scale", "auto_deprecated"],
                                        default="auto_deprecated")
            tol_space = LogFloatSpace(name="tol", min_val=1e-4, max_val=1e-2, default=1e-3)
            C_space = LogFloatSpace(name="C", min_val=1e-2, max_val=1e2, default=1.0)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            parameter_space.merge([kernel_space,
                                   nu_space,
                                   degree_space,
                                   gamma_space,
                                   tol_space,
                                   C_space,
                                   shrinking_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class LinearSVR(BaseRegressor):

    def __init__(self,
                 epsilon=0.0,
                 tol=1e-4,
                 C=1.0,
                 loss='epsilon_insensitive',
                 fit_intercept=True,
                 intercept_scaling=1.,
                 dual=True,
                 verbose=0,
                 random_state=None,
                 max_iter=1000,
                 e_id=None,
                 **kwargs
                 ):
        super(LinearSVR, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

        self.model_name = 'Linear_SVR'
        self.model = sklearn.svm.LinearSVR(tol=self.tol,
                                           C=self.C,
                                           epsilon=self.epsilon,
                                           fit_intercept=self.fit_intercept,
                                           intercept_scaling=self.intercept_scaling,
                                           verbose=self.verbose,
                                           random_state=self.random_state,
                                           max_iter=self.max_iter,
                                           dual=self.dual,
                                           loss=self.loss)

    @property
    def coef_(self):
        return self.coef_

    @property
    def intercept_(self):
        return self.intercept_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-1, default=1e-4)
            C_space = LogFloatSpace(name='C', min_val=1e-2, max_val=1e2, default=1)
            loss_space = CategorySpace(name="loss", choice_space=["epsilon_insensitive", "squared_epsilon_insensitive"],
                                       default="epsilon_insensitive")
            fit_intercept_space = CategorySpace(name="fit_intercept", choice_space=[True, False], default=True)
            dual_space = CategorySpace(name="dual", choice_space=[True, False], default=True)
            parameter_space.merge([tol_space,
                                   C_space,
                                   loss_space,
                                   fit_intercept_space,
                                   dual_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
