import sklearn
from das.ParameterSpace import *
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor


class ARDRegression(BaseRegressor):
    def __init__(self,
                 n_iter=300,
                 tol=1.e-3,
                 alpha_1=1.e-6,
                 alpha_2=1.e-6,
                 lambda_1=1.e-6,
                 lambda_2=1.e-6,
                 compute_score=False,
                 threshold_lambda=1.e+4,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 verbose=False,
                 e_id=None,
                 random_state=None,
                 **kwargs
                 ):
        super(ARDRegression, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

        self.model_name = "ARDRegression"
        self.model = sklearn.linear_model.ARDRegression(n_iter=self.n_iter,
                                                        tol=self.tol,
                                                        alpha_1=self.alpha_1,
                                                        alpha_2=self.alpha_2,
                                                        lambda_1=self.lambda_1,
                                                        lambda_2=self.lambda_2,
                                                        compute_score=self.compute_score,
                                                        threshold_lambda=self.threshold_lambda,
                                                        fit_intercept=self.fit_intercept,
                                                        normalize=self.normalize,
                                                        copy_X=self.copy_X,
                                                        verbose=self.verbose)

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def alpha_(self):
        return self.model.alpha_

    @property
    def lambda_(self):
        return self.model.lambda_

    @property
    def sigma_(self):
        return self.sigma_

    @property
    def scores_(self):
        return self.model.scores_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            tol_space = LogFloatSpace(name="tol", min_val=10**-5, max_val=0.1, default=10**-3)
            alpha_1_space = UniformFloatSpace(name="alpha_1", min_val=10**-10, max_val=10**-3, default=10**-6)
            alpha_2_space = LogFloatSpace(name="alpha_2", min_val=10**-10, max_val=10**-3, default=10**-6)
            lambda_1_space = LogFloatSpace(name="lambda_1", min_val=10**-10, max_val=10**-3, default=10**-6)
            lambda_2_space = LogFloatSpace(name="lambda_2", min_val=10**-10, max_val=10**-3, default=10**-6)
            threshold_lambda_space = LogFloatSpace(name="threshold_lambda", min_val=10**3, max_val=10**5, default=10**4)

            parameter_space.merge([
                tol_space,
                alpha_1_space,
                alpha_2_space,
                lambda_1_space,
                lambda_2_space,
                threshold_lambda_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
