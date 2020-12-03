from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn.svm import SVC as sk_svc
from sklearn.svm import NuSVC as sk_nusvc
from sklearn.svm import LinearSVC as sk_linearsvc
from das.ParameterSpace import *


class SVC(BaseClassifier):

    """
    the fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset
    with more than a couple of 10000 samples
    """

    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=True,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None,
                 e_id=None,
                 **kwargs):
        super(SVC, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        
        self.model_name = "SVC"
        self.model = sk_svc(C=self.C,
                            kernel=self.kernel,
                            degree=self.degree,
                            gamma=self.gamma,
                            coef0=self.coef0,
                            shrinking=self.shrinking,
                            probability=self.probability,
                            tol=self.tol,
                            cache_size=self.cache_size,
                            class_weight=self.class_weight,
                            verbose=self.verbose,
                            max_iter=self.max_iter,
                            decision_function_shape=self.decision_function_shape,
                            random_state=self.random_state)
        self.reward = None

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            C_space = UniformFloatSpace(name="C", min_val=0, max_val=1e5, default=1.0)
            kernel_space = CategorySpace(name="kernel", choice_space=["linear", "poly", "rbf", "sigmoid"],
                                         default="rbf")
            degree_space = UniformIntSpace(name="degree", min_val=1, max_val=5, default=3)
            gamma_space = LogFloatSpace(name="gamma", min_val=1e-5, max_val=1, default=0.1)
            coef0_space = UniformFloatSpace(name="coef0", min_val=-1, max_val=1, default=0)
            # probability_space = CategorySpace(name="probability", choice_space=[True, False], default=False)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-3)

            parameter_space.merge([C_space,
                                   kernel_space,
                                   degree_space,
                                   gamma_space,
                                   coef0_space,
                                   # probability_space,
                                   shrinking_space,
                                   # tol_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class NuSVC(BaseClassifier):
    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=True,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None,
                 e_id=None,
                 **kwargs):
        super(NuSVC, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        
        self.model_name = "NuSVC"
        self.model = sk_nusvc(nu=self.nu,
                              kernel=self.kernel,
                              degree=self.degree,
                              gamma=self.gamma,
                              coef0=self.coef0,
                              shrinking=self.shrinking,
                              probability=self.probability,
                              tol=self.tol,
                              cache_size=self.cache_size,
                              class_weight=self.class_weight,
                              verbose=self.verbose,
                              max_iter=self.max_iter,
                              decision_function_shape=self.decision_function_shape,
                              random_state=self.random_state)

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            # nu_space = UniformFloatSpace(name="nu", min_val=1e-9, max_val=1, default=0.5)
            kernel_space = CategorySpace(name="kernel", choice_space=["linear", "poly", "rbf", "sigmoid"],
                                         default="rbf")
            degree_space = UniformIntSpace(name="degree", min_val=1, max_val=5, default=3)
            gamma_space = LogFloatSpace(name="gamma", min_val=1e-5, max_val=1, default=0.1)
            coef0_space = UniformFloatSpace(name="coef0", min_val=-1, max_val=1, default=0)
            # probability_space = CategorySpace(name="probability", choice_space=[True, False], default=False)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-3)

            parameter_space.merge([
                kernel_space,
                degree_space,
                gamma_space,
                coef0_space,
                # probability_space,
                shrinking_space,
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class LinearSVC(BaseClassifier):
    def __init__(self,
                 penalty='l2',
                 loss='squared_hinge',
                 dual=True,
                 tol=1e-4,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=1000,
                 e_id=None,
                 **kwargs):
        super(LinearSVC, self).__init__(e_id=e_id, random_state=random_state, **kwargs)

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

        self.model_name = "LinearSVC"
        self.model = sk_linearsvc(penalty=self.penalty,
                                  loss=self.loss,
                                  dual=self.dual,
                                  tol=self.tol,
                                  C=self.C,
                                  multi_class=self.multi_class,
                                  fit_intercept=self.fit_intercept,
                                  intercept_scaling=self.intercept_scaling,
                                  class_weight=self.class_weight,
                                  verbose=self.verbose,
                                  random_state=self.random_state,
                                  max_iter=self.max_iter)

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2"], default="l2")
            loss_space = CategorySpace(name="loss", choice_space=["hinge", "squared_hinge"], default="squared_hinge")
            C_space = UniformFloatSpace(name="C", min_val=0, max_val=1e5, default=1.0)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-3)

            parameter_space.merge([penalty_space,
                                   loss_space,
                                   C_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
