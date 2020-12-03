from das.BaseAlgorithm.Classification import LGBClassifier
from das.BaseAlgorithm.Classification import XGBClassifier
from das.BaseAlgorithm.Classification import SGDClassifier
from das.BaseAlgorithm.Classification import LogisticRegression
from das.BaseAlgorithm.Classification import IdentityClassifier
from das.BaseAlgorithm.Classification import TreeClassifier as TreeClassifier
from das.BaseAlgorithm.Classification import RandomForestClassifier as RandomForestClassifier
from das.BaseAlgorithm.Classification import SVM as SVM_Classifier
from das.BaseAlgorithm.Classification import Adaboost as AdaboostClassifier
from das.BaseAlgorithm.Classification import DA
from das.BaseAlgorithm.Classification import ExtraTreesClassifier
from das.BaseAlgorithm.Classification import GBDT as GBDTClassifier
from das.BaseAlgorithm.Classification import GPClassifier
from das.BaseAlgorithm.Classification import MLPClassifier as MLPClassifier
from das.BaseAlgorithm.Classification import NN
from das.BaseAlgorithm.Classification import NB

from das.BaseAlgorithm.Regression import LGBRegressor
from das.BaseAlgorithm.Regression import XGBRegressor
from das.BaseAlgorithm.Regression import Adaboost as AdaboostRegressor
from das.BaseAlgorithm.Regression import ARDRegression
from das.BaseAlgorithm.Regression import ExtraTreesRegressor
from das.BaseAlgorithm.Regression import GBDT as GBDTRegressor
from das.BaseAlgorithm.Regression import GPRegressor as GPR
from das.BaseAlgorithm.Regression import NN as NNRegression
from das.BaseAlgorithm.Regression import RandomForestRegressor as RandomForestRegressor
from das.BaseAlgorithm.Regression import Ridge
from das.BaseAlgorithm.Regression import SGDRegressor
from das.BaseAlgorithm.Regression import SVR
from das.BaseAlgorithm.Regression import TreeRegressor as TreeRegressor
from das.BaseAlgorithm.Regression import IdentityRegressor as IdentityRegressor

import warnings
from copy import deepcopy
from das.pipeline import Pipeline
warnings.filterwarnings("ignore")

# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import Binarizer
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import MaxAbsScaler
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import MinMaxScaler
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import Normalizer
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import OneHotEncoder
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import PolynomialFeatures
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import QuantileTransformer
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import RobustScaler
# from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import StandardScaler
#
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import FactorAnalysis
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import FastICA
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import KernelPCA
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import LDA
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import NMF
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import PCA
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import RFE
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import SelectFromModel
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import SVD
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import UnivariateFeatureSelection
# from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import VarianceThreshhold
#
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import DBSCAN
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import KMeans, MiniBatchKMeans
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import AffinityPropagation
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import AgglomerativeClustering
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import Birch
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import GaussianMixture
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import MeanShift
# from das.BaseAlgorithm.Clustering.BaseAlgorithm import SpectralClustering

__all_datapreprocess_algorithm_keys = ["Binarizer",
                                       "MaxAbsScaler",
                                       "MinMaxScaler",
                                       "Normalizer",
                                       "OneHotEncoder",
                                       "PolynomialFeatures",
                                       "QuantileTransformer",
                                       "RobustScaler",
                                       "StandardScaler"]

__all_featureengineering_algorithm_keys = ["FactorAnalysis",
                                           "FastICA",
                                           "KernelPCA",
                                           "LDA",
                                           "NMF",
                                           "PCA",
                                           "RFE",
                                           "SelectFromModel",
                                           "SVD",
                                           "SelectKBest",
                                           "SelectPercentile",
                                           "SelectFpr",
                                           "SelectFdr",
                                           "SelectFwe"]

__all_classification_algorithm_keys = ["SGDClassifier",
                                       "GBDTClassifier",
                                       "LogisticRegression",
                                       "DecisionTreeClassifier",
                                       "RandomForestClassifier",
                                       "SVC",
                                       "AdaboostClassifier",
                                       "LinearDiscriminantAnalysis",
                                       "QuadraticDiscriminantAnalysis",
                                       "ExtraTreesClassifier",
                                       "BernoulliNB",
                                       "MultinomialNB",
                                       "GaussianNB",
                                       # "GPClassifier",
                                       "MLPClassifier",
                                       "KNeighborsClassifier",
                                       # "RadiusNeighborsClassifier",
                                       "XGBClassifier",
                                       "LGBClassifier",
                                       # "IdentityClassifier"
                                       ]

__all_regression_algorithm_keys = [
    "AdaboostRegressor",
    "ARDRegression",
    "ExtraTreesRegressor",
    "GBDTRegressor",
    # "GPRegressor",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "RandomForestRegressor",
    "Ridge",
    "SGDRegressor",
    "epsilon_svr",
    # "nu_svr",
    "linear_svr",
    "DecisionTreeRegressor",
    "XGBRegressor",
    "LGBRegressor",
    # "IdentityRegressor"
]

__all_clustering_algorthm_keys = ["AffinityPropagation",
                                  "AgglomerativeClustering",
                                  "Birch",
                                  "DBSCAN",
                                  "GaussianMixture",
                                  "KMeans",
                                  "MeanShift",
                                  "MiniBatchKMeans",
                                  "SpectralClustering"]


def judge_algorithm_key(key):
    if key in __all_classification_algorithm_keys:
        return 'classification'
    elif key in __all_regression_algorithm_keys:
        return 'regression'
    elif key in __all_clustering_algorthm_keys:
        return 'clustering'
    else:
        raise "Not supported key: {}".format(key)


def get_all_datapreprocess_algorithm_keys():
    return deepcopy(__all_datapreprocess_algorithm_keys)


def get_all_featureengineering_algorithm_keys():
    return deepcopy(__all_featureengineering_algorithm_keys)


def get_all_classification_algorithm_keys():
    return deepcopy(__all_classification_algorithm_keys)


def get_all_regression_algorithm_keys():
    return deepcopy(__all_regression_algorithm_keys)


def get_algorithm_key_dict(algo_space=None):
    all_algos = algo_space
    if all_algos is None:
        all_algos = get_all_classification_algorithm_keys()
    ret_dict = {}
    for i, algo in enumerate(all_algos):
        ret_dict[algo] = i
    return ret_dict


def get_all_clustering_algorithm_keys():
    return deepcopy(__all_clustering_algorthm_keys)


# def __get_datapreprocess_algorithm(key, random_state=None):
#     if key == "Binarizer":
#         return Binarizer.Binarizer()
#     elif key == "MaxAbsScaler":
#         return MaxAbsScaler.MaxAbsScaler()
#     elif key == "MinMaxScaler":
#         return MinMaxScaler.MinMaxScaler()
#     elif key == "Normalizer":
#         return Normalizer.Normalizer()
#     elif key == "OneHotEncoder":
#         return OneHotEncoder.OneHotEncoder()
#     elif key == "PolynomialFeatures":
#         return PolynomialFeatures.PolynomialFeatures()
#     elif key == "QuantileTransformer":
#         raise NotImplementedError
#     elif key == "RobustScaler":
#         return RobustScaler.RobustScaler()
#     elif key == "StandardScaler":
#         return StandardScaler.StandardScaler()
#
#
# def __get_featureengineering_algorithm(key, random_state=None):
#     if key == "FactorAnalysis":
#         return FactorAnalysis.FactorAnalysis(random_state=random_state)
#     elif key == "FastICA":
#         return FastICA.FastICA(random_state=random_state)
#     elif key == "KernelPCA":
#         return KernelPCA.KernelPCA(random_state=random_state)
#     elif key == "LDA":
#         return LDA.LatentDirichletAllocation(random_state=random_state)
#     elif key == "NMF":
#         return NMF.NMF(random_state=random_state)
#     elif key == "PCA":
#         return PCA.PCA(random_state=random_state)
#     elif key == "RFE":
#         return RFE.RFE()
#     elif key == "SelectFromModel":
#         # return SelectFromModel.SelectFromModel()
#         return SelectFromModel.SelectFromModel()
#     elif key == "SVD":
#         return SVD.SVD(random_state=random_state)
#     elif key == "SelectKBest":
#         return UnivariateFeatureSelection.SelectKBest()
#     elif key == "SelectPercentile":
#         return UnivariateFeatureSelection.SelectPercentile()
#     elif key == "SelectFpr":
#         return UnivariateFeatureSelection.SelectFpr()
#     elif key == "SelectFdr":
#         return UnivariateFeatureSelection.SelectFdr()
#     elif key == "SelectFwe":
#         return UnivariateFeatureSelection.SelectFwe()


def __get_classification_algorithm(key, random_state=None):
    if key == "SGDClassifier":
        return SGDClassifier.SGDClassifier(random_state=random_state)
    elif key == "GBDTClassifier":
        return GBDTClassifier.GBDTClassifier(random_state=random_state)
    elif key == "LogisticRegression":
        return LogisticRegression.LogisticRegression(random_state=random_state)
    elif key == "DecisionTreeClassifier":
        return TreeClassifier.DecisionTreeClassifier(random_state=random_state)
    elif key == "RandomForestClassifier":
        return RandomForestClassifier.RandomForestClassifier(random_state=random_state)
    elif key == "SVC":
        return SVM_Classifier.SVC(random_state=random_state)
    elif key == "AdaboostClassifier":
        return AdaboostClassifier.AdaBoostClassifier(random_state=random_state)
    elif key == "LinearDiscriminantAnalysis":
        return DA.LinearDiscriminantAnalysis()
    elif key == "QuadraticDiscriminantAnalysis":
        return DA.QuadraticDiscriminantAnalysis()
    elif key == "ExtraTreesClassifier":
        return ExtraTreesClassifier.ExtraTreesClassifier(random_state=random_state)
    elif key == "BernoulliNB":
        return NB.BernoulliNB()
    elif key == "MultinomialNB":
        return NB.MultinomialNB()
    elif key == "GaussianNB":
        return NB.GaussianNB()
    elif key == "GPClassifier":
        return GPClassifier.GPClassifier(random_state=random_state)
    elif key == "MLPClassifier":
        return MLPClassifier.MLPClassifier(random_state=random_state)
    elif key == "KNeighborsClassifier":
        return NN.KNeighborsClassifier()
    elif key == "RadiusNeighborsClassifier":
        return NN.RadiusNeighborsClassifier()
    elif key == 'XGBClassifier':
        return XGBClassifier.XGBClassifier(random_state=random_state)
    elif key == 'LGBClassifier':
        return LGBClassifier.LGBClassifier(random_state=random_state)
    elif key == 'IdentityClassifier':
        return IdentityClassifier.IdentityClassifier(random_state=random_state)


def __get_regression_algorithm(key, random_state=None):
    if key == "AdaboostRegressor":
        return AdaboostRegressor.AdaboostRegressor(random_state=random_state)
    elif key == "ARDRegression":
        return ARDRegression.ARDRegression()
    elif key == "ExtraTreesRegressor":
        return ExtraTreesRegressor.ExtraTreesRegressor(random_state=random_state)
    elif key == "GBDTRegressor":
        return GBDTRegressor.GBDTRegressor(random_state=random_state)
    elif key == 'XGBRegressor':
        return XGBRegressor.XGBRegressor(random_state=random_state)
    elif key == "GPRegressor":
        return GPR.GPRegressor(random_state=random_state)
    elif key == "KNeighborsRegressor":
        return NNRegression.KNeighborsRegressor()
    elif key == "RadiusNeighborsRegressor":
        return NNRegression.RadiusNeighborsRegressor()
    elif key == "RandomForestRegressor":
        return RandomForestRegressor.RandomForestRegressor(random_state=random_state)
    elif key == "Ridge":
        return Ridge.Ridge(random_state=random_state)
    elif key == "SGDRegressor":
        return SGDRegressor.SGDRegressor(random_state=random_state)
    elif key == "epsilon_svr":
        return SVR.SVR()
    elif key == "nu_svr":
        return SVR.NuSVR()
    elif key == "linear_svr":
        return SVR.LinearSVR(random_state=random_state)
    elif key == "DecisionTreeRegressor":
        return TreeRegressor.DecisionTreeRegressor(random_state=random_state)
    elif key == 'LGBRegressor':
        return LGBRegressor.LGBRegressor(random_state=random_state)
    elif key == 'IdentityRegressor':
        return IdentityRegressor.IdentityRegressor(random_state=random_state)


def __get_classification_algorithm_class(key):
    obj = __get_classification_algorithm(key, 0)
    return obj.__class__


def __get_regression_algorithm_class(key):
    obj = __get_regression_algorithm(key, 0)
    return obj.__class__

# def __get_clustering_algorithnm(key, random_state=None):
#     if key == "AffinityPropagation":
#         return AffinityPropagation.AffinityPropagation()
#     elif key == "AgglomerativeClustering":
#         return AgglomerativeClustering.AgglomerativeClustering()
#     elif key == "Birch":
#         return Birch.Birch()
#     elif key == "DBSCAN":
#         return DBSCAN.DBSCAN()
#     elif key == "GaussianMixture":
#         return GaussianMixture.GaussianMixture(random_state=random_state)
#     elif key == "KMeans":
#         return KMeans.KMeans(random_state=random_state)
#     elif key == "MeanShift":
#         return MeanShift.MeanShift()
#     elif key == "MiniBatchKMeans":
#         return MiniBatchKMeans.MiniBatchKMeans(random_state=random_state)
#     elif key == "SpectralClustering":
#         return SpectralClustering.SpectralClustering(random_state=random_state)


def get_algorithm_by_key(key, random_state=None):
    if key in __all_classification_algorithm_keys:
        return __get_classification_algorithm(key, random_state)
    elif key in __all_regression_algorithm_keys:
        return __get_regression_algorithm(key, random_state)
    else:
        raise Exception("系统中没有该算法", key)


def get_algorithm_class_by_key(key):
    if key in __all_classification_algorithm_keys:
        return __get_classification_algorithm_class(key)
    elif key in __all_regression_algorithm_keys:
        return __get_regression_algorithm_class(key)
    else:
        raise Exception("系统中没有该算法", key)

# def get_algorithm_by_key(key, random_state=None):
#     if __validate_algorithm_key(key):
#         raise Exception("系统中没有该算法", key)
#
#     if key in __all_featureengineering_algorithm_keys:
#         return __get_featureengineering_algorithm(key, random_state)
#     elif key in __all_datapreprocess_algorithm_keys:
#         return __get_datapreprocess_algorithm(key, random_state)
#     elif key in __all_Classification_algorithm_keys:
#         return __get_Classification_algorithm(key, random_state)
#     elif key in __all_regression_algorithm_keys:
#         return __get_regression_algorithm(key, random_state)
#     elif key in __all_clustering_algorthm_keys:
#         return __get_clustering_algorithnm(key, random_state)
#     elif key == 'IdentityRegressor':
#         return __get_regression_algorithm(key, random_state)


def get_pipeline_by_key(key):
    pipeline = []
    for k in key:
        pipeline.append((k, get_algorithm_by_key(k)))
    pipeline = Pipeline(pipeline)
    return pipeline


def construct_algorithm_set_by_keys(keys, random_state=None):
    """
    根据算法名称的集合来构建算法集合
    """
    res = {}
    for i in keys:
        res[i] = get_algorithm_by_key(i, random_state=random_state)
    return res


if __name__ == "__main__":
    name = "ExtraTreesClassifier"
    obj = get_algorithm_by_key(name, 0)
    class_obj = get_algorithm_class_by_key(name)
    print(obj, class_obj)

