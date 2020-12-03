import sklearn
from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor


class Ridge(BaseRegressor):

    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 max_iter=None,
                 tol=1e-3,
                 solver="auto",
                 random_state=None,
                 e_id=None,
                 **kwargs):
        super(Ridge, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

        self.model_name = "Ridge"
        self.model = sklearn.linear_model.Ridge(alpha=self.alpha,
                                                fit_intercept=self.fit_intercept,
                                                normalize=self.normalize,
                                                copy_X=self.copy_X,
                                                max_iter=self.max_iter,
                                                tol=self.tol,
                                                solver=self.solver,
                                                random_state=self.random_state)

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def n_iter_(self):
        return self.model.n_iter_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            alpha_space = LogFloatSpace(name="alpha", min_val=10**-5, max_val=10., default=1.)
            tol_space = LogFloatSpace(name="tol", min_val=1e-5, max_val=1e-1, default=1e-3)
            parameter_space.merge([
                alpha_space,
                tol_space
            ])

        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
