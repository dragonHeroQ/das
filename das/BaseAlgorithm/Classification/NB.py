from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.ParameterSpace import *
from das.util.decorators import check_model
from sklearn.naive_bayes import BernoulliNB as sk_bernoulliNB
from sklearn.naive_bayes import GaussianNB as sk_gaussianNB
from sklearn.naive_bayes import MultinomialNB as sk_multinomialNB


class GaussianNB(BaseClassifier):
    def __init__(self, priors=None, e_id=None, random_state=None, **kwargs):
        super(GaussianNB, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.priors = priors
        self.model_name = "GaussianNB"
        self.model = sk_gaussianNB(priors=self.priors, **kwargs)

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
            if self.model is None:
                raise Exception
            parameter_space = ParameterSpace()

        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def partial_fit(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y, sample_weight=sample_weight)
        return self


class MultinomialNB(BaseClassifier):
    def __init__(self,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(MultinomialNB, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.model_name = "MultinomialNB"
        self.model = sk_multinomialNB(alpha=self.alpha,
                                      class_prior=self.class_prior,
                                      fit_prior=self.fit_prior)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()

            alpha_space = UniformFloatSpace(name="alpha", min_val=0, max_val=2.0, default=1.0)
            fit_prior_space = CategorySpace(name="fit_prior", choice_space=[True, False], default=True)

            parameter_space.merge([alpha_space, fit_prior_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class BernoulliNB(BaseClassifier):
    def __init__(self,
                 alpha=1.0,
                 binarize=0.0,
                 fit_prior=True,
                 class_prior=None,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(BernoulliNB, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.binarize = binarize

        self.model_name = "BernoulliNB"
        self.model = sk_bernoulliNB(alpha=self.alpha,
                                    binarize=self.binarize,
                                    class_prior=self.class_prior,
                                    fit_prior=self.fit_prior)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:

            alpha_space = UniformFloatSpace(name="alpha", min_val=0, max_val=2.0, default=1.0)

            fit_prior = CategorySpace(name="fit_prior", choice_space=[True, False], default=True)

            parameter_space.merge([alpha_space, fit_prior])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        return self


if __name__ == "__main__":
    import sklearn.datasets
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    m = BernoulliNB()
    m.fit(X, y)
    print(m.predict_proba(X))
