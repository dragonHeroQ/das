import das
import logging
import numpy as np
from das.performance_evaluation import eval_performance, judge_rule

logger = logging.getLogger(das.logger_name)


def aggregate_probas_to_proba(y_pred, n_classes):
	if y_pred is None:
		return None
	logger.debug("y_pred.shape = {}, n_classes = {}".format(y_pred.shape, n_classes))
	assert (n_classes > 0 and y_pred.shape[1] % n_classes == 0), \
		("Y_pred.shape[1]={} not equal to k*n_classes=k*{}".format(y_pred.shape[1], n_classes))
	n_components = y_pred.shape[1] // n_classes
	logger.debug("n_components = {}".format(n_components))
	n_example = y_pred.shape[0]
	final_proba = np.zeros((n_example, n_classes), dtype=y_pred.dtype)
	for i in range(n_components):
		final_proba += y_pred[:, i * n_classes:(i + 1) * n_classes]
	final_proba /= n_components
	logger.debug("final proba and shape = {}".format(final_proba.shape if final_proba is not None else None))
	return final_proba


# def aggregate_probas_to_proba(y_pred, y_follow_pred, n_classes):
# 	logger.debug("y_pred.shape = {}, n_classes = {}".format(y_pred.shape, n_classes))
# 	assert (n_classes > 0 and y_pred.shape[1] % n_classes == 0),\
# 		("Y_pred.shape[1]={} not equal to k*n_classes=k*{}".format(y_pred.shape[1], n_classes))
# 	n_components = y_pred.shape[1] // n_classes
# 	logger.debug("n_components = {}".format(n_components))
# 	n_train = y_pred.shape[0]
# 	final_proba = np.zeros((n_train, n_classes), dtype=y_pred.dtype)
# 	final_follow_proba = None
# 	if y_follow_pred is not None:
# 		final_follow_proba = np.zeros((y_follow_pred.shape[0], n_classes), dtype=y_follow_pred.dtype)
# 	for i in range(n_components):
# 		final_proba += y_pred[:, i * n_classes:(i + 1) * n_classes]
# 		if y_follow_pred is not None:
# 			final_follow_proba += y_follow_pred[:, i * n_classes:(i + 1) * n_classes]
# 	return final_proba, final_follow_proba


def aggregate_probas_to_prediction(y_pred, n_classes, task='classification'):
	if y_pred is None:
		return None
	logger.debug("y_pred.shape={}, n_classes={}".format(y_pred.shape, n_classes))
	final_proba = aggregate_probas_to_proba(y_pred, n_classes)
	if task == 'classification':
		final_prediction = np.argmax(final_proba, axis=1)
	else:
		final_prediction = np.mean(final_proba, axis=1)
	logger.debug("final prediction: {}".format(final_prediction.shape if final_prediction is not None else None))
	return final_prediction


# def aggregate_probas_to_prediction(y_pred, y_follow_pred, n_classes):
# 	final_proba = aggregate_probas_to_proba(y_pred, n_classes)
# 	final_follow_proba = aggregate_probas_to_proba(y_follow_pred, n_classes)
# 	final_prediction = np.argmax(final_proba, axis=1)
# 	final_follow_prediction = None
# 	if y_follow_pred is not None:
# 		final_follow_prediction = np.argmax(final_follow_proba, axis=1)
# 	return final_prediction, final_follow_prediction


def from_probas_to_performance(y_pred, y, n_classes, task='classification', evaluation_rule=None, classes_=None):
	if y_pred is None:
		return None
	logger.debug("task = {}".format(task))
	logger.debug("n_classes = {}".format(n_classes))
	final_prediction = aggregate_probas_to_prediction(y_pred, n_classes, task=task)
	if task == 'classification':
		assert classes_ is not None, "classes_ should not be None"
		final_prediction = np.array(list(map(lambda x: classes_[x], final_prediction)))
	performance = eval_performance(rule=evaluation_rule, y_true=y, y_score=final_prediction)
	return performance


# def from_probas_to_performance(y_pred, y, y_follow_pred, y_follow, n_classes, evaluation_rule='accuracy_score'):
# 	final_prediction = aggregate_probas_to_prediction(y_pred, n_classes)
# 	train_performance = eval_performance(rule=evaluation_rule, y_true=y, y_score=final_prediction)
# 	if y_follow_pred is not None:
# 		final_follow_prediction = aggregate_probas_to_prediction(y_follow_pred, n_classes)
# 		follow_performance = eval_performance(rule=evaluation_rule, y_true=y_follow, y_score=final_follow_prediction)
# 	else:
# 		follow_performance = None
# 	return train_performance, follow_performance


def infer_n_classes(y, evaluation_rule):
	n_classes = None
	if judge_rule(evaluation_rule) == 'classification':
		n_classes = len(np.unique(y))
	return n_classes

