from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from sklearn.dummy import DummyRegressor as sk_DummyRegressor


class DummyRegressor(BaseRegressor):
    """
    DummyRegressor is a regressor that makes predictions using
    simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0]
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : float or array of shape [n_outputs]
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int,
        Number of outputs.

    outputs_2d_ : bool,
        True if the output at fit is 2d, else false.
    """
    def __init__(self,
                 e_id=None,
                 random_state=None):
        super(DummyRegressor, self).__init__(e_id=e_id, random_state=random_state)

        self.model_name = "DummyRegressor"
        self.model = sk_DummyRegressor()

