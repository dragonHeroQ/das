from sklearn.neural_network import MLPRegressor as sk_mlpregressor
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.ParameterSpace import *


class MLPRegressor(BaseRegressor):
	"""Multi-layer Perceptron regressor.

    This model optimizes the squared-loss using LBFGS or stochastic gradient
    descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, optional, default 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when solver='sgd'.

    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, optional, default 0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    warm_start : bool, optional, default False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default 0.9
        Momentum for gradient descent update.  Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : boolean, default True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    epsilon : float, optional, default 1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

        .. versionadded:: 0.20

    Attributes
    ----------
    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : string
        Name of the output activation function.

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
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
		super(MLPRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
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
		self.model = sk_mlpregressor(hidden_layer_sizes=self.hidden_layer_sizes,
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
			solver_space = CategorySpace(name="solver", choice_space=["lbfgs", "sgd", "adam"], default='adam')
			activation_space = CategorySpace(name='activation',
			                                 choice_space=['identity', 'logistic', 'tanh', 'relu'], default='relu')
			alpha_space = LogFloatSpace(name='alpha', min_val=0.0001, max_val=0.01, default=0.0001)
			learning_rate_space = CategorySpace(name='learning_rate',
			                                    choice_space=['constant', 'invscaling', 'adaptive'], default='constant')
			parameter_space.merge([solver_space,
			                       activation_space,
			                       alpha_space,
			                       learning_rate_space])
		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space
