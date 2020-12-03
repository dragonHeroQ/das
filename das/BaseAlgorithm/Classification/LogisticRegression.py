from sklearn import linear_model
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.ParameterSpace import *


class LogisticRegression(BaseClassifier):

    def __init__(self,
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=True,
                 class_weight=None,
                 random_state=None,
                 solver='liblinear',
                 max_iter=100,
                 multi_class='ovr',
                 verbose=0,
                 warm_start=False,
                 n_jobs=-1,
                 e_id=None,
                 **kwargs):
        super(LogisticRegression, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        self.model_name = "LogisticRegression"
        self.model = linear_model.LogisticRegression(penalty=self.penalty,
                                                     dual=self.dual,
                                                     tol=self.tol,
                                                     C=self.C,
                                                     fit_intercept=self.fit_intercept,
                                                     intercept_scaling=self.intercept_scaling,
                                                     class_weight=self.class_weight,
                                                     random_state=self.random_state,
                                                     solver=self.solver,
                                                     max_iter=self.max_iter,
                                                     multi_class=self.multi_class,
                                                     verbose=self.verbose,
                                                     warm_start=self.warm_start,
                                                     n_jobs=self.n_jobs)

    def set_configuration_space(self, ps=None):
        """

        :param ps: dict类型
        :return:
        """
        parameter_space = ParameterSpace()
        if ps is None:
            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2"], default="l2")
            # dual fromulation is only implemented for l2 penalty with liblinear solver.
            # prefer dual=False when n_samples > n_features
            dual_space = CategorySpace(name="dual", choice_space=[False], default=False)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-4)
            C_space = UniformFloatSpace(name="C", min_val=1e-4, max_val=1e4, default=1.0)
            # fit_intercept_space = CategorySpace([True, False], default=True)
            max_iter_space = UniformIntSpace(name="max_iter", min_val=50, max_val=500, default=100)
            parameter_space.merge([penalty_space,
                                   dual_space,
                                   max_iter_space,
                                   # tol_space,
                                   C_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
