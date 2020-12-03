from sklearn import linear_model
from das.ParameterSpace import *
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier


class SGDClassifier(BaseClassifier):
    def __init__(self,
                 loss='log',
                 penalty='l2',
                 alpha=0.0001,
                 l1_ratio=0.15,
                 fit_intercept=True,
                 max_iter=None,
                 tol=None,
                 shuffle=True,
                 verbose=0,
                 epsilon=0.1,
                 n_jobs=-1,
                 random_state=None,
                 learning_rate='optimal',
                 eta0=0.0,
                 power_t=0.5,
                 class_weight=None,
                 warm_start=False,
                 average=False,
                 n_iter=None,
                 e_id=None,
                 **kwargs):
        super(SGDClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
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
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        self.n_iter = n_iter

        self.model_name = "SGDClassifier"
        self.model = linear_model.SGDClassifier(loss=self.loss,
                                                penalty=self.penalty,
                                                alpha=self.alpha,
                                                l1_ratio=self.l1_ratio,
                                                fit_intercept=self.fit_intercept,
                                                max_iter=self.max_iter,
                                                tol=self.tol,
                                                shuffle=self.shuffle,
                                                verbose=self.verbose,
                                                epsilon=self.epsilon,
                                                n_jobs=self.n_jobs,
                                                random_state=self.random_state,
                                                learning_rate=self.learning_rate,
                                                eta0=self.eta0,
                                                power_t=self.power_t,
                                                class_weight=self.class_weight,
                                                warm_start=self.warm_start,
                                                average=self.average,
                                                n_iter=self.n_iter,
                                                **kwargs)

    def partial_fit(self, X, y):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y)
        return self

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            parameter_space = ParameterSpace()
            # Note: hinge loss does not support predict_proba~~~~
            # loss_space = CategorySpace(name="loss",
            #                            choice_space=["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            #                            default="hinge")
            loss_space = CategorySpace(name="loss",
                                       choice_space=["log", "modified_huber"],
                                       default="log")
            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2", "elasticnet", "none"], default="l2")
            alpha_space = LogFloatSpace(name="alpha", min_val=1e-6, max_val=1e-2, default=1e-4)
            l1_ratio_space = LogFloatSpace(name="l1_ratio", min_val=1e-6, max_val=1, default=0.15)
            max_iter_space = UniformIntSpace(name="max_iter", min_val=5, max_val=20000, default=1000)
            # shuffle_space = CategorySpace(name="shuffle", choice_space=[True, False], default=True)
            learning_rate_space = CategorySpace(name="learning_rate",
                                                choice_space=["constant", "optimal", "invscaling"], default="optimal")
            eta0_space = LogFloatSpace(name="eta0", min_val=1e-6, max_val=1e-1, default=0.01)
            power_t_space = UniformFloatSpace(name="power_t", min_val=1e-5, max_val=1, default=0.5)
            average_space = CategorySpace(name="average", choice_space=[True, False], default=False)

            parameter_space.merge([loss_space,
                                   penalty_space,
                                   alpha_space,
                                   l1_ratio_space,
                                   max_iter_space,
                                   # tol_space,
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
