from sklearn.metrics import *
from das.util.decorators import deprecated
import numpy as np

# TODO: record the worst metric and best metric of each rule.
#  For example, "accuracy_score": {"max_val": 1.0, "min_val": 0.0}
#  so that we can easily get the corresponding metrics.
# TODO: try to re-construct, using OOP (big)
# TODO: try to check the correctness of these max_vals and min_vals
__classification_rule_set = ["accuracy_score",
                             "auc",
                             "average_precision_score",
                             "balanced_accuracy_score",
                             "brier_score_loss",
                             "classification_report",
                             "cohen_kappa_score",
                             "confusion_matrix",
                             "f1_score",
                             "fbeta_score",
                             "hamming_loss",
                             "hinge_loss",
                             "jaccard_similarity_score",
                             "log_loss",
                             "matthews_corrcoef",
                             "precision_recall_curve",
                             "precision_recall_fscore_support",
                             "precision_score",
                             "recall_score",
                             "roc_auc_score",
                             "roc_curve",
                             "zero_one_loss",
                             # Multi-label
                             # TODO(fang xin): coverage_error, The best value is equal to the average
                             #  number of labels in y_true per sample.
                             "coverage_error",
                             "label_ranking_average_precision_score",
                             "label_ranking_loss",
                             "top_1_accuracy",
                             "top_1_precision"]

__larger_is_better_classification_rule_set = [
    "accuracy_score",
    "auc",
    "average_precision_score",
    "balanced_accuracy_score",
    "cohen_kappa_score",
    "f1_score",
    "fbeta_score",
    "jaccard_similarity_score",
    "matthews_corrcoef",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "label_ranking_average_precision_score",
    "top_1_accuracy",
    "top_1_precision"
]

__smaller_is_better_classfication_rule_set = [
    "brier_score_loss",
    "label_ranking_loss",
    "hamming_loss",
    "hinge_loss",
    "log_loss",
    "zero_one_loss",
    "label_ranking_loss"
]

__larger_is_better_clustering_rule_set = [
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "completeness_score",
    "fowlkes_mallows_score",
    "homogeneity_completeness_v_score",
    "homogeneity_score",
    "mutual_info_score",
    "normalized_mutual_info_score",
    "v_measure_score",
    "silhouette_score",
    "calinski_harabaz_score",
]

__smaller_is_better_clustering_rule_set = [
    "davies_bouldin_score"
]

__larger_is_better_regression_rule_set = [
    "explained_variance_score",
    "r2_score"
]

__smaller_is_better_regression_rule_set = [
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "median_absolute_error"
]

__clustering_rule_set = ["adjusted_mutual_info_score",
                         "adjusted_rand_score",
                         "completeness_score",
                         "fowlkes_mallows_score",
                         "homogeneity_score",
                         "mutual_info_score",
                         "normalized_mutual_info_score",
                         "v_measure_score",

                         # 不需要真实label
                         "silhouette_score",
                         "calinski_harabaz_score",
                         "davies_bouldin_score"]

__regression_rule_set = ["explained_variance_score",
                         "mean_absolute_error",
                         "mean_squared_error",
                         "mean_squared_log_error",
                         "median_absolute_error",
                         "r2_score"]


def get_classification_rule_set():
    return __classification_rule_set


def get_clustering_rule_set():
    return __clustering_rule_set


def get_regression_rule_set():
    return __regression_rule_set


def get_larger_is_better_classification_rule_set():
    return __larger_is_better_classification_rule_set


def get_smaller_is_better_classification_rule_set():
    return __smaller_is_better_classfication_rule_set


def get_larger_is_better_regression_rule_set():
    return __larger_is_better_regression_rule_set


def get_smaller_is_better_regression_rule_set():
    return __smaller_is_better_regression_rule_set


def get_larger_is_better_clustering_set():
    return __larger_is_better_clustering_rule_set


def get_smaller_is_better_clustering_set():
    return __smaller_is_better_clustering_rule_set


def judge_rule(rule):
    if rule in get_classification_rule_set():
        return "classification"
    elif rule in get_regression_rule_set():
        return "regression"
    elif rule in get_clustering_rule_set():
        return "clustering"
    raise Exception("Unknown evaluation_rule={}".format(rule))


def eval_performance_classification(rule="accuracy_score", y_true=None, y_score=None, normalize=True,
                                    average='micro', pos_label=1, labels=None, sample_weight=None,
                                    reorder="deprecated", x=None, y=None):
    if rule == "accuracy_score":
        return accuracy_score(y_true, y_score, normalize=normalize, sample_weight=sample_weight)
    elif rule == "balanced_accuracy_score":
        return balanced_accuracy_score(y_true, y_score, sample_weight=sample_weight)
    elif rule == "auc":
        return auc(x=x, y=y, reorder=reorder)
    elif rule == "average_precision_score":
        return average_precision_score(y_true, y_score, average=average, pos_label=pos_label,
                                       sample_weight=sample_weight)
    elif rule == "brier_score_loss":
        return brier_score_loss(y_true, y_score, sample_weight=sample_weight, pos_label=pos_label)
    elif rule == "classification_report":
        # TODO(fang xin): not a number, returns string / dict
        # return classification_report(y_true, y_score)
        raise NotImplementedError
    elif rule == "cohen_kappa_score":
        # float (-1, 1), The maximum value mean complete agreement; zero or lower means chance agreement
        return cohen_kappa_score(y_true, y_score)
    elif rule == "cofusion_matrix":
        # TODO(fang xin): not a number
        # array, shape = [n_classes, n_classes]
        # return confusion_matrix(y_true, y_score)
        raise NotImplementedError
    elif rule == "f1_score":
        return f1_score(y_true, y_score)
        # raise NotImplementedError
    elif rule == "fbeta_score":
        # TODO(fang xin): fbeta_score need hyper parameter
        raise NotImplementedError
    elif rule == "hamming_loss":
        return hamming_loss(y_true=y_true, y_pred=y_score, labels=labels, sample_weight=sample_weight)
    elif rule == "hinge_loss":
        return hinge_loss(y_true, y_score)
        # raise NotImplementedError
    elif rule == "jaccard_similarity_score":
        return jaccard_similarity_score(y_true, y_score)
        # raise NotImplementedError
    elif rule == "log_loss":
        return log_loss(y_true, y_score, eps=0.01, normalize=normalize, sample_weight=sample_weight, labels=labels)
    elif rule == "matthews_corrcoef":
        # between -1 and +1, A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1
        # an inverse prediction
        return matthews_corrcoef(y_true, y_score, sample_weight=sample_weight)
        # raise NotImplementedError
    elif rule == "precision_recall_curve":
        # TODO(fang xin): not a number, returns precision, recall, thresholds
        raise NotImplementedError
    elif rule == "precision_recall_fscore_support":
        # TODO(fang xin): not a number, returns precision, recall, fbeta_score, support
        raise NotImplementedError
    elif rule == "precision_score":
        return precision_score(y_true, y_score, labels=labels, pos_label=pos_label, average=average,
                               sample_weight=sample_weight)
    elif rule == "recall_score":
        return recall_score(y_true, y_score, labels, pos_label=pos_label, average=average, sample_weight=sample_weight)
    elif rule == "roc_auc_score":
        return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight)
    elif rule == "roc_auc_curve":
        # TODO(fang xin): not a number, returns fpr, tpr, thresholds
        raise NotImplementedError
    elif rule == "zero_one_loss":
        # If normalize == True, return the fraction of misclassifications (float),
        # else it returns the number of misclassifications (int)
        return zero_one_loss(y_true, y_score, normalize=True)
        # raise NotImplementedError
    elif rule == "top_1_accuracy":
        return top_1_accuracy(y_true=y_true, y_pred=y_score)
    elif rule == "top_1_precision":
        return top_1_precision(y_true=y_true, y_pred=y_score)

    # multi label
    elif rule == "coverage_error":
        return coverage_error(y_true=y_true, y_score=y_score, sample_weight=sample_weight)
    elif rule == "label_ranking_average_precision_score":
        return label_ranking_average_precision_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight)
    elif rule == "label_ranking_loss":
        return label_ranking_loss(y_true=y_true, y_score=y_score, sample_weight=sample_weight)


def eval_performance_regression(rule="explained_variance_score", y_true=None, y_score=None, sample_weight=None,
                                multioutput="uniform_average"):
    if rule == "explained_variance_score":
        return explained_variance_score(y_true, y_score, sample_weight=sample_weight, multioutput=multioutput)
    elif rule == "mean_absolute_error":
        return mean_absolute_error(y_true, y_score, sample_weight=sample_weight, multioutput=multioutput)
    elif rule == "mean_squared_error":
        return mean_squared_error(y_true, y_score, sample_weight=sample_weight, multioutput=multioutput)
    elif rule == "mean_squared_log_error":
        return mean_squared_log_error(y_true, y_score, sample_weight=sample_weight, multioutput=multioutput)
    elif rule == "median_absolute_error":
        return median_absolute_error(y_true, y_score)
    elif rule == "r2_score":
        return r2_score(y_true, y_score, sample_weight=sample_weight, multioutput=multioutput)


def eval_performance_clustering(rule="adjusted_mutual_info_score", labels_true=None, labels_pred=None,
                                average_method="warn", sparse=False, contingency=None, X=None, metric="euclidean",
                                sample_size=None, random_state=None, eps=None):
    """
    需要样本真实标记的评估策略：adjusted_mutual_info_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, fowlkes_mallows_score, v_measure_score, Contigency Matrix
    不需要样本真实标记的评估策略： Silhouette Coefficient, Calinski-Harabaz Index, Davies-Bouldin Index

    :param rule:
    :param labels_true:
    :param labels_pred:
    :param average_method:
    :param sparse:
    :param contingency:
    :return:
    """
    # 需要真实标记
    if rule == "adjusted_mutual_info_score":
        return adjusted_mutual_info_score(labels_true, labels_pred, average_method=average_method)
    elif rule == "adjusted_rand_score":
        return adjusted_rand_score(labels_true, labels_pred)
    elif rule == "completeness_score":
        return completeness_score(labels_true, labels_pred)
    elif rule == "fowlkes_mallows_score":
        return fowlkes_mallows_score(labels_true, labels_pred, sparse=sparse)
    elif rule == "homogeneity_completeness_v_score":
        # TODO(fang xin): not a number, return (homogeneity, completeness, v_measure), the v_measure is harmonic mean of the first two
        raise NotImplementedError
    elif rule == "homogeneity_score":
        return homogeneity_score(labels_true, labels_pred)
    elif rule == "mutual_info_score":
        return mutual_info_score(labels_true, labels_pred, contingency=contingency)
    elif rule == "normalized_mutual_info_score":
        return normalized_mutual_info_score(labels_true, labels_pred, average_method=average_method)
    elif rule == "v_measure_score":
        return v_measure_score(labels_true, labels_pred)
    elif rule == "contingency_matrix":
        # TODO(fang xin): not a number, returns a matrix
        # return contingency_matrix(labels_true=labels_true, labels_pred=labels_pred, eps=eps, sparse=sparse)
        raise NotImplementedError
    # 不需要真实标记
    if rule == "silhouette_score":
        return silhouette_score(X=X, labels=labels_pred, sample_size=sample_size,
                                metric=metric, random_state=random_state)
    elif rule == "calinski_harabaz_score":
        return calinski_harabaz_score(X=X, labels=labels_pred)
    elif rule == "davies_bouldin_score":
        return davies_bouldin_score(X=X, labels=labels_pred)


def eval_performance(rule="accuracy_score",
                     y_true=None,
                     y_score=None,
                     normalize=True,
                     average="macro",
                     pos_label=1,
                     labels=None,
                     sample_weight=None,
                     average_method="warn",
                     multioutput="uniform_average",
                     sparse=False,
                     contingency=None,
                     sample_size=None,
                     X=None,
                     metric="euclidean",
                     random_state=None):
    if y_score is None:
        return None
    cg = judge_rule(rule)
    if cg == "classification":
        # TODO: remove this to relieve overhead!
        # print("y_true = {}, y_score = {}".format(type(y_true), type(y_score)))
        # print(np.unique(y_true), np.unique(y_score), type(np.unique(y_true) == np.unique(y_score)))
        # assert (np.unique(y_true)[0] == np.unique(y_score)[0]), (
        #     "y_true contains {}, but y_score contains {}".format(np.unique(y_true), np.unique(y_score)))
        return eval_performance_classification(rule=rule,
                                               y_true=y_true,
                                               y_score=y_score,
                                               normalize=normalize,
                                               average=average,
                                               pos_label=pos_label,
                                               labels=labels,
                                               sample_weight=sample_weight)
    elif cg == "regression":
        return eval_performance_regression(rule=rule,
                                           y_true=y_true,
                                           y_score=y_score,
                                           sample_weight=sample_weight,
                                           multioutput=multioutput
                                           )
    elif cg == "clustering":
        return eval_performance_clustering(rule=rule,
                                           labels_true=y_true,
                                           labels_pred=y_score,
                                           average_method=average_method,
                                           sparse=sparse,
                                           contingency=contingency,
                                           X=X,
                                           sample_size=sample_size,
                                           metric=metric,
                                           random_state=random_state)

    else:
        raise Exception("无效的评估策略")


def _comp_v1(a, b):
    """
    larger is better
    """
    if a >= b:
        return True
    else:
        return False


def _comp_v2(a, b):
    """
    smaller is better
    """
    if a <= b:
        return True
    else:
        return False


def compare_performance(rule, a, b):
    cg = judge_rule(rule)
    if cg == "classification":
        if rule in get_larger_is_better_classification_rule_set():
            return _comp_v1(a, b)
        else:
            return _comp_v2(a, b)

    elif cg == "regression":
        if rule in get_larger_is_better_regression_rule_set():
            return _comp_v1(a, b)
        else:
            return _comp_v2(a, b)

    elif cg == "clustering":
        if rule in get_larger_is_better_clustering_set():
            return _comp_v1(a, b)
        else:
            return _comp_v2(a, b)

    else:
        raise Exception("Invalid evaluation rule")


@deprecated
def initial_min_val(rule):
    if is_larger_better(rule):
        return -np.Inf
    return 0


def get_min_value(rule):
    if rule in ['accuracy_score', 'roc_auc_score']:
        return 0.0
    if rule in ['mean_squared_error']:
        return 1e82
    if rule in ['r2_score']:
        return 0.0
    return None


def initial_worst_score(rule):
    if is_larger_better(rule):
        min_val = get_min_value(rule)
        if min_val is None:
            return -1e82
        return min_val
    return 1e82


def initial_worst_loss(rule):
    # loss, always the larger the better
    if get_min_value(rule) is not None:
        return score_to_loss(rule, initial_worst_score(rule))
    return 1e82


def is_larger_better(rule):
    if (rule in get_larger_is_better_classification_rule_set() or
        rule in get_larger_is_better_regression_rule_set() or
            rule in get_larger_is_better_clustering_set()):
        return True
    return False


def score_to_loss(rule, score):
    if rule in ["accuracy_score", "auc", "balanced_accuracy_score", "f1_score",
                "precision_score", "recall_score", "roc_auc_score", 'r2_score']:
        return 1.0-score
    return -score if is_larger_better(rule) else score


def loss_to_score(rule, loss):
    if rule in ["accuracy_score", "auc", "balanced_accuracy_score", "f1_score",
                "precision_score", "recall_score", "roc_auc_score", 'r2_score']:
        return 1.0-loss
    return -loss if is_larger_better(rule) else loss


def compare_and_update(rule, score1, score2):
    if is_larger_better(rule):
        if score1 > score2:
            return score2
    else:
        if score1 < score2:
            return score2
    return score1


def top_1_accuracy(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true_label = np.argmax(y_true, axis=-1)
    else:
        y_true_label = y_true
    if len(y_pred.shape) == 2:
        y_pred_label = np.argmax(y_pred, axis=-1)
    else:
        y_pred_label = y_pred
    return accuracy_score(y_true=y_true_label, y_pred=y_pred_label)


@deprecated
def top_1_precision(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=-1)
    return accuracy_score(y_true=y_true, y_pred=y_pred_label)


if __name__ == "__main__":
    print(initial_worst_loss('accuracy_score'))
