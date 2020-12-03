from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from sklearn.dummy import DummyClassifier as sk_DummyClassifier


class DummyClassifier(BaseClassifier):
    """
    DummyClassifier is a classifier that makes predictions using simple rules.

    This classifier is useful as a simple baseline to compare with other
    (real) classifiers. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str, default="stratified"
        Strategy to use to generate predictions.

        * "stratified": generates predictions by respecting the training
          set's class distribution.
        * "most_frequent": always predicts the most frequent label in the
          training set.
        * "prior": always predicts the class that maximizes the class prior
          (like "most_frequent") and ``predict_proba`` returns the class prior.
        * "uniform": generates predictions uniformly at random.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class

          .. versionadded:: 0.17
             Dummy Classifier now supports prior fitting strategy using
             parameter *prior*.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    constant : int or str or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    """
    def __init__(self,
                 e_id=None,
                 random_state=None):
        super(DummyClassifier, self).__init__(e_id=e_id, random_state=random_state)

        self.model_name = "DummyClassifier"
        self.model = sk_DummyClassifier(random_state=self.random_state)


if __name__ == "__main__":
    pass
