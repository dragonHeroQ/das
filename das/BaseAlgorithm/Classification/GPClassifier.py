import traceback
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.ParameterSpace import *
from sklearn.gaussian_process.kernels import *
from das.util.decorators import check_model
from sklearn.gaussian_process import GaussianProcessClassifier


class GPClassifier(BaseClassifier):
    def __init__(self,
                 kernel=None,
                 optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0,
                 max_iter_predict=100,
                 warm_start=False,
                 copy_X_train=True,
                 random_state=None,
                 multi_class='one_vs_rest',
                 n_jobs=-1,
                 e_id=None,
                 **kwargs):
        super(GPClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        
        self.rbf_thetaL = None
        self.rbf_thetaU = None

        self.model_name = "GPClassifier"
        self.model = GaussianProcessClassifier(kernel=self.kernel,
                                               optimizer=self.optimizer,
                                               n_restarts_optimizer=self.n_restarts_optimizer,
                                               max_iter_predict=self.max_iter_predict,
                                               warm_start=self.warm_start,
                                               copy_X_train=self.copy_X_train,
                                               random_state=self.random_state,
                                               multi_class=self.multi_class,
                                               n_jobs=self.n_jobs)

    @check_model
    def fit(self, X, y, **fit_params):
        n_features = X.shape[1]
        ker = RBF(length_scale=[1.0] * n_features,
                  length_scale_bounds=[(self.rbf_thetaL, self.rbf_thetaU)] * n_features)
        self.model.kernel = ker
        self.kernel = ker
        try:
            self.model = self.model.fit(X, y)
        except Exception as e:
            print(e)
        return self.model

    def predict_proba(self, X, **predict_params):
        ret = None
        try:
            ret = self.model.predict_proba(X)
        except Exception as e:
            print(e)
        finally:
            return ret

    @check_model
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        return self.model.log_marginal_likelihood(theta=theta, eval_gradient=eval_gradient)

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            rbf_thetaL_space = LogFloatSpace(name='rbf_thetaL', min_val=1e-10, max_val=1e-3, default=1e-6)
            setattr(self.model, 'rbf_thetaL', 1e-6)
            rbf_thetaU_space = LogFloatSpace(name='rbf_thetaU', min_val=1.0, max_val=100000, default=100000.0)
            setattr(self.model, 'rbf_thetaU', 100000.0)
            n_restarts_optimizer_space = UniformIntSpace(name="n_restarts_optimizer", min_val=0, max_val=20, default=0)

            parameter_space.merge([
                rbf_thetaL_space,
                rbf_thetaU_space,
                n_restarts_optimizer_space,
                ]
            )
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

