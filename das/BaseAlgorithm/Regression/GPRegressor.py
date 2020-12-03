from das.util.decorators import check_model
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from das.ParameterSpace import *
from sklearn.gaussian_process.kernels import *


class GPRegressor(BaseRegressor):

    def __init__(self,
                 kernel=None,
                 alpha=1e-10,
                 optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 normalize_y=False,
                 copy_X_train=True,
                 random_state=None,
                 rbf_thetaL=1e-6,
                 rbf_thetaU=100000.0,
                 e_id=None,
                 ):
        super(GPRegressor, self).__init__(e_id=e_id, random_state=random_state)
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

        self.rbf_thetaL = rbf_thetaL
        self.rbf_thetaU = rbf_thetaU

        self.model_name = "GPRegressor"
        self.model = GaussianProcessRegressor(kernel=self.kernel,
                                              alpha=self.alpha,
                                              optimizer=self.optimizer,
                                              n_restarts_optimizer=self.n_restarts_optimizer,
                                              normalize_y=self.normalize_y,
                                              copy_X_train=self.copy_X_train,
                                              random_state=self.random_state
                                              )

    @check_model
    def fit(self, X, y, **fit_params):
        n_features = X.shape[1]
        ker = RBF(length_scale=[1.0]*n_features, length_scale_bounds=[(self.rbf_thetaL, self.rbf_thetaU)]*n_features)
        self.model.kernel = ker
        self.kernel = ker
        return self.model.fit(X, y)

    @check_model
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        return self.model.log_marginal_likelihood(theta=theta, eval_gradient=eval_gradient)

    @check_model
    def sample_y(self, X, n_samples=1, random_state=0):
        return self.model.sample_y(X, n_samples=n_samples, random_state=random_state)

    @property
    def X_train_(self):
        return self.model.X_train_

    @property
    def y_train_(self):
        return self.model.y_train_

    @property
    def kernel_(self):
        return self.model.kernel_

    @property
    def L_(self):
        return self.model.L_

    @property
    def alpha_(self):
        return self.model.alpha_

    @property
    def log_marginal_likelihood_value_(self):
        return self.model.log_marginal_likelihood_value_

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            alpha_space = LogFloatSpace(name='alpha', min_val=1e-14, max_val=1.0, default=1e-8)
            rbf_thetaL_space = LogFloatSpace(name='rbf_thetaL', min_val=1e-10, max_val=1e-3, default=1e-6)
            setattr(self.model, 'rbf_thetaL', 1e-6)
            rbf_thetaU_space = LogFloatSpace(name='rbf_thetaU', min_val=1.0, max_val=100000, default=100000.0)
            setattr(self.model, 'rbf_thetaU', 100000.0)
            n_restarts_optimizer_space = UniformIntSpace(name="n_restarts_optimizer", min_val=0, max_val=20, default=0)
            normalize_y_space = CategorySpace(name="normalize_y", choice_space=[True, False], default=False)

            parameter_space.merge([
                                   alpha_space,
                                   rbf_thetaL_space,
                                   rbf_thetaU_space,
                                   n_restarts_optimizer_space,
                                   normalize_y_space]
                                  )
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


if __name__ == '__main__':
    g = GPRegressor(kernel=RBF(1.0))
    print(g.get_params(deep=True))
