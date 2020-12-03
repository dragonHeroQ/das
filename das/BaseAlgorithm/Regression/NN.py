import sklearn
from das.BaseAlgorithm.Regression.BaseAlgorithm import BaseRegressor
from das.ParameterSpace import *
from das.util.decorators import check_model


class KNeighborsRegressor(BaseRegressor):
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
        super(KNeighborsRegressor, self).__init__(e_id=e_id, random_state=random_state, **kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model_name = "KNeighborsRegressor"
        self.model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                           weights=self.weights,
                                                           algorithm=self.algorithm,
                                                           leaf_size=self.leaf_size,
                                                           p=self.p,
                                                           metric=self.metric,
                                                           metric_params=self.metric_params,
                                                           n_jobs=self.n_jobs,
                                                           **kwargs)

    @check_model
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        return self.model.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance)

    @check_model
    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        return self.model.kneighbors_graph(X=X, n_neighbors=n_neighbors, mode=mode)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            n_neighbors_space = LogIntSpace(name="n_neighbors", min_val=1, max_val=100, default=1)
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            algorithm_space = CategorySpace(name="algorithm", choice_space=["auto", "ball_tree", "kd_tree", "brute"],
                                            default="auto")
            p_space = CategorySpace(name="p", choice_space=[1, 2], default=2)

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


class RadiusNeighborsRegressor(BaseRegressor):

    def __init__(self,
                 radius=1.0,
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
        super(RadiusNeighborsRegressor, self).__init__(e_id=e_id, random_state=random_state)
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model_name = "RadiusNeighborsRegressor"
        self.model = sklearn.neighbors.RadiusNeighborsRegressor(radius=self.radius,
                                                                weights=self.weights,
                                                                algorithm=self.algorithm,
                                                                leaf_size=self.leaf_size,
                                                                p=self.p,
                                                                metric=self.metric,
                                                                metric_params=self.metric_params,
                                                                n_jobs=self.n_jobs,
                                                                **kwargs)

    @check_model
    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        return self.model.radius_neighbors(X=X, radius=radius, return_distance=return_distance)

    @check_model
    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        return self.model.radius_neighbors_graph(X, radius=radius, mode=mode)

    @check_model
    def _with_e_id_changed(self):
        pass

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            algorithm_space = CategorySpace(name="algorithm", choice_space=["ball_tree", "kd_tree", "brute", "auto"],
                                            default="auto")
            p_space = CategorySpace(name="p", choice_space=[1, 2], default=2)

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
