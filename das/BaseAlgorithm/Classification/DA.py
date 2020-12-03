from sklearn import discriminant_analysis
from das.ParameterSpace import *
from das.util.decorators import check_model
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier


class LinearDiscriminantAnalysis(BaseClassifier):

    def __init__(self,
                 solver='svd',
                 shrinkage=None,
                 priors=None,
                 n_components=None,
                 store_covariance=False,
                 tol=0.0001,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(LinearDiscriminantAnalysis, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

        self.model_name = "LinearDA"
        self.model = discriminant_analysis.LinearDiscriminantAnalysis(solver=self.solver,
                                                                      shrinkage=self.shrinkage,
                                                                      priors=self.priors,
                                                                      n_components=self.n_components,
                                                                      store_covariance=self.n_components,
                                                                      tol=self.tol)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()
            solver_space = CategorySpace(name="solver", choice_space=["svd", "lsqr", "eigen"], default="svd")
            # TODO: shrinkage parameter, possible values:
            # None: no shrinkage(default)
            # 'auto': automatic shrinkage using the Ledoit-Wolf lemma
            # float between 0 and 1: fixed shrinkage parameter.
            # 2018.11.20 QIU changed None to 1.0
            shrinkage_space = CategorySpace(name="shrinkage", choice_space=["auto", 1.0], default=1.0)
            parameter_space.merge([solver_space, shrinkage_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)

    def fit_transform(self, X):
        if self.model is None:
            raise Exception
        return self.model.fit_transform(X)

    def transform(self, X):
        if self.model is None:
            raise Exception
        return self.model.transform(X)


class QuadraticDiscriminantAnalysis(BaseClassifier):

    def __init__(self,
                 priors=None,
                 reg_param=0.0,
                 store_covariance=False,
                 tol=0.0001,
                 store_covariances=None,
                 e_id=None,
                 random_state=None,
                 **kwargs):

        super(QuadraticDiscriminantAnalysis, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.store_covariances = store_covariances
        self.tol = tol

        self.model_name = "QuadraticDA"
        self.model = discriminant_analysis.QuadraticDiscriminantAnalysis(priors=self.priors,
                                                                         reg_param=self.reg_param,
                                                                         store_covariance=self.store_covariance,
                                                                         tol=self.tol,
                                                                         store_covariances=self.store_covariances)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        """

        :param ps: dict类型
        :return:
        """
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()

            # priors_space = CategorySpace(choice_space=[None], default=None)
            reg_param_space = UniformFloatSpace(name="reg_param", min_val=0.0, max_val=1.0, default=0.0)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-4)

            parameter_space.merge([reg_param_space,
                                   # tol_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)

