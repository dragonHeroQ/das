from sklearn.neural_network import MLPClassifier as sk_mlpclassifier
from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.ParameterSpace import *


class MLPClassifier(BaseClassifier):

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=0.0001,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 e_id=None,
                 **kwargs):
        super(MLPClassifier, self).__init__(e_id=e_id, random_state=random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.model_name = "MLPClassifier"
        self.model = sk_mlpclassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                      activation=self.activation,
                                      solver=self.solver,
                                      alpha=self.alpha,
                                      batch_size=self.batch_size,
                                      learning_rate=self.learning_rate,
                                      learning_rate_init=self.learning_rate_init,
                                      power_t=self.power_t,
                                      max_iter=self.max_iter,
                                      shuffle=self.shuffle,
                                      random_state=self.random_state,
                                      tol=self.tol,
                                      verbose=self.verbose,
                                      warm_start=self.warm_start,
                                      momentum=self.momentum,
                                      nesterovs_momentum=self.nesterovs_momentum,
                                      early_stopping=self.early_stopping,
                                      validation_fraction=self.validation_fraction,
                                      beta_1=self.beta_1,
                                      epsilon=self.epsilon,
                                      **kwargs)

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            solver_space = CategorySpace(name="solver", choice_space=["lbfgs", "sgd", "adam"])
            activation_space = CategorySpace(name='activation',
                                             choice_space=['identity', 'logistic', 'tanh', 'relu'], default='relu')
            alpha_space = LogFloatSpace(name='alpha', min_val=0.0001, max_val=0.01, default=0.0001)
            learning_rate_space = CategorySpace(name='learning_rate',
                                                choice_space=['constant', 'invscaling', 'adaptive'], default='constant')
            parameter_space.merge([solver_space,
                                   activation_space,
                                   alpha_space,
                                   learning_rate_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
