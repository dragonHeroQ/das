from das.BaseAlgorithm.Classification.BaseAlgorithm import BaseClassifier
from das.ParameterSpace import *
from das.util.decorators import check_model
from sklearn.neighbors import KNeighborsClassifier as sk_kneighborsclassifier
from sklearn.neighbors import RadiusNeighborsClassifier as sk_radiusneighborsclassifier


class KNeighborsClassifier(BaseClassifier):
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=-1,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(KNeighborsClassifier, self).__init__(e_id=e_id, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model_name = "KNeighborClassifier"
        self.model = sk_kneighborsclassifier(n_neighbors=self.n_neighbors,
                                             weights=self.weights,
                                             algorithm=self.algorithm,
                                             leaf_size=self.leaf_size,
                                             p=self.p,
                                             metric=self.metric,
                                             metric_params=self.metric_params,
                                             n_jobs=self.n_jobs,
                                             **kwargs)

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
            n_neighbors_space = UniformIntSpace(name="n_neighbors", min_val=1, max_val=20, default=5)
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            algorithm_space = CategorySpace(name="algorithm", choice_space=["auto", "ball_tree", "kd_tree", "brute"],
                                            default="auto")
            p_space = UniformIntSpace(name="p", min_val=1, max_val=5, default=2)

            parameter_space.merge([n_neighbors_space,
                                   weights_space,
                                   algorithm_space,
                                   p_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space


class RadiusNeighborsClassifier(BaseClassifier):

    def __init__(self,
                 radius=1.0,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 outlier_label=None,
                 metric_params=None,
                 n_jobs=-1,
                 e_id=None,
                 random_state=None,
                 **kwargs):
        super(RadiusNeighborsClassifier, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.outlier_label = outlier_label
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model_name = "RadiusNeighborsClassifier"
        self.model = sk_radiusneighborsclassifier(radius=self.radius,
                                                  weights=self.weights,
                                                  algorithm=self.algorithm,
                                                  leaf_size=self.leaf_size,
                                                  p=self.p,
                                                  metric=self.metric,
                                                  outlier_label=self.outlier_label,
                                                  metric_params=self.metric_params,
                                                  n_jobs=self.n_jobs,
                                                  **kwargs)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            algorithm_space = CategorySpace(name="algorithm", choice_space=["BallTree", "KDTree", "brute", "auto"],
                                            default="auto")
            p_space = UniformIntSpace(name="p", min_val=1, max_val=5, default=2)

            parameter_space.merge([weights_space,
                                   algorithm_space,
                                   p_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
