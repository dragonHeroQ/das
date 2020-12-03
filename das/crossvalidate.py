import das
import ray
import copy
import time
import logging
import traceback
import numpy as np
from joblib import delayed, Parallel
from das.util.common_utils import kill_tree
from das.performance_evaluation import *
from sklearn.model_selection import StratifiedKFold, KFold

logger = logging.getLogger(das.logger_name)


# TODO: add try catch finally to handle the exception in fit stage
def cross_validate_score(model, X, y, cv=5, predict_method="predict",
                         evaluation_rule="accuracy_score", random_state=None):
	if len(y.shape) == 1:
		y_unique = len(set(y))
	else:
		y_unique = y.shape[1]

	row_n = X.shape[0]
	col_n = X.shape[1]

	if predict_method == "predict" and len(y.shape) == 1:
		y_final_val_pred = np.zeros((row_n,))
	else:
		y_final_val_pred = np.zeros((row_n, y_unique))

	scores = []
	if len(y.shape) == 1 and judge_rule(evaluation_rule) != "regression":
		logger.debug("start cv, StratifiedKFold")
		splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
	else:
		logger.debug("start cv, KFold")
		splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
	fold_idx = 0
	for train_index, test_index in splitter.split(X, y):

		# start_time = time.time()
		clf = model
		X_train = X[train_index]
		y_train = y[train_index]
		X_val = X[test_index]
		y_val = y[test_index]

		clf.fit(X_train, y_train)

		if predict_method == "predict":
			y_val_pred = clf.predict(X_val)
		elif predict_method == "predict_proba":
			y_val_pred = clf.predict_proba(X_val)
		else:
			raise Exception("invalid predict method")

		y_val_pred = np.nan_to_num(y_val_pred)
		y_final_val_pred[test_index] = y_val_pred
		score = eval_performance(evaluation_rule, y_true=y_val, y_score=y_val_pred, random_state=random_state)
		scores.append(score)
		# print("Fold {} cost {:.2f} s".format(fold_idx, time.time()-start_time))
		fold_idx += 1

	return np.mean(scores), y_final_val_pred


def _fit_and_score(model, X, y, train_idx, val_idx, predict_method="predict",
                   evaluation_rule="accuracy_score", random_state=None):
	clf = copy.deepcopy(model)
	X_train = X[train_idx]
	y_train = y[train_idx]
	X_val = X[val_idx]
	y_val = y[val_idx]
	clf.fit(X_train, y_train)

	if predict_method == "predict":
		y_val_pred = clf.predict(X_val)
	elif predict_method == "predict_proba":
		y_val_pred = clf.predict_proba(X_val)
	else:
		raise Exception("invalid predict method")

	y_val_pred = np.nan_to_num(y_val_pred)
	score = eval_performance(evaluation_rule, y_true=y_val, y_score=y_val_pred, random_state=random_state)
	return score


def parallel_cross_validate_score(model, X, y, cv=5, predict_method="predict", n_jobs=-1, verbose=0,
                                  pre_dispatch='2*n_jobs', evaluation_rule="accuracy_score", random_state=None):
	if len(y.shape) == 1:
		y_unique = len(set(y))
	else:
		y_unique = y.shape[1]

	row_n = X.shape[0]
	col_n = X.shape[1]
	# TODO: fill up y_final_val_pred
	if predict_method == "predict" and len(y.shape) == 1:
		y_final_val_pred = np.zeros((row_n,))
	else:
		y_final_val_pred = np.zeros((row_n, y_unique))

	if len(y.shape) == 1 and judge_rule(evaluation_rule) != "regression":
		logger.debug("start cv, StratifiedKFold")
		splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
	else:
		logger.debug("start cv, KFold")
		splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

	parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
	# start_time = time.time()
	scores = parallel(
		delayed(_fit_and_score)(
			model, X, y, train_idx, val_idx, predict_method, evaluation_rule, random_state)
		for train_idx, val_idx in splitter.split(X, y))
	# print("CV {} cost {:.2f} s".format(cv, time.time() - start_time))

	return np.mean(scores), y_final_val_pred


def _predict_proxy(estimator, X, predict_method='predict_proba'):
	if X is None:
		return None
	return getattr(estimator, predict_method)(X)


def _fit_predict_proxy(estimator, X, y, X_follows=None, predict_method='predict_proba', distribute=0, **kwargs):
	if X is None or X_follows is None:  # nothing to fit or nothing to predict
		return None
	return estimator.fit_predict(X=X, y=y, X_follows=X_follows,
	                             predict_method=predict_method, distribute=distribute, **kwargs)


def _regularize_output(y_pred):
	if y_pred is None:
		return None
	return np.nan_to_num(y_pred)


def _fit_and_transform(model, X, y, train_idx, val_idx, X_follow=None, predict_method="predict_proba"):
	clf = copy.deepcopy(model)
	X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]
	clf.fit(X_train, y_train)

	y_val_pred = _predict_proxy(estimator=clf, X=X_val, predict_method=predict_method)
	y_follow_pred = _predict_proxy(estimator=clf, X=X_follow, predict_method=predict_method)

	y_val_pred, y_follow_pred = _regularize_output(y_val_pred), _regularize_output(y_follow_pred)

	return y_val_pred, y_follow_pred


def fit_and_transform_v2(model, X, y, train_idx, val_idx, X_follow=None,
                         predict_method="predict_proba", distribute=0, **kwargs):
	"""
	使用 fit predict 接口，方便并行化，无需先 fit 然后再 predict

	:param model:
	:param X:
	:param y:
	:param train_idx:
	:param val_idx:
	:param X_follow:
	:param predict_method:
	:return:
	"""
	clf = copy.deepcopy(model)
	X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]

	y_val_pred, y_follow_pred = _fit_predict_proxy(estimator=clf, X=X_train, y=y_train,
	                                               X_follows=(X_val, X_follow), predict_method=predict_method,
	                                               distribute=distribute, **kwargs)

	y_val_pred, y_follow_pred = _regularize_output(y_val_pred), _regularize_output(y_follow_pred)

	return y_val_pred, y_follow_pred


def parallel_cross_validate_transform(model, X, y, X_follow=None, cv=5, predict_method="predict_proba",
                                      n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                                      task="classification", random_state=None, **kwargs):
	row_n = X.shape[0]
	y_final_val_pred = None
	y_final_follow_pred = None
	if len(y.shape) == 1 and task != "regression":
		logger.debug("start cv, StratifiedKFold")
		splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
	else:
		logger.debug("start cv, KFold")
		splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

	parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
	# start_time = time.time()
	train_val_idx = list(splitter.split(X, y))
	predictions = parallel(
		delayed(_fit_and_transform)(
			model=model, X=X, y=y, train_idx=train_idx, val_idx=val_idx,
			X_follow=X_follow, predict_method=predict_method)
		for train_idx, val_idx in train_val_idx)
	# print("CV {} cost {:.2f} s".format(cv, time.time() - start_time))
	for i, (y_val_pred, y_follow_pred) in enumerate(predictions):
		logger.debug("y_val_pred.shape={}, y_follow_pred.shape={}".format(
			y_val_pred.shape, y_follow_pred.shape if y_follow_pred is not None else None))
		if i == 0:
			shape_1 = y_val_pred.shape[1]
			y_final_val_pred = np.zeros((row_n, shape_1))
			if X_follow is not None:
				row_follow = X_follow.shape[0]
				y_final_follow_pred = np.zeros((row_follow, shape_1))
		y_final_val_pred[train_val_idx[i][1]] = y_val_pred
		if X_follow is not None:
			y_final_follow_pred += y_follow_pred
	if X_follow is not None:
		y_final_follow_pred /= cv

	return y_final_val_pred, y_final_follow_pred


def seq_cross_validate_transform(model, X, y, X_follow=None, cv=5, predict_method="predict_proba",
                                 task="classification", random_state=None, **kwargs):
	row_n = X.shape[0]
	y_final_val_pred = None
	y_final_follow_pred = None
	if len(y.shape) == 1 and task != "regression":
		logger.debug("start cv, StratifiedKFold")
		splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
	else:
		logger.debug("start cv, KFold")
		splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

	train_val_idx = list(splitter.split(X, y))

	distribute = 0
	if 'distribute' in kwargs:
		distribute = kwargs['distribute']
		kwargs.pop('distribute')

	predictions = []
	for train_idx, val_idx in train_val_idx:
		predictions.append(fit_and_transform_v2(
			model=model, X=X, y=y, train_idx=train_idx, val_idx=val_idx, X_follow=X_follow,
			predict_method=predict_method, distribute=distribute, **kwargs))

	for i, (y_val_pred, y_follow_pred) in enumerate(predictions):
		logger.debug("y_val_pred.shape={}, y_follow_pred.shape={}".format(
			y_val_pred.shape, y_follow_pred.shape if y_follow_pred is not None else None))
		if i == 0:
			shape_1 = y_val_pred.shape[1]
			y_final_val_pred = np.zeros((row_n, shape_1))
			if X_follow is not None:
				row_follow = X_follow.shape[0]
				y_final_follow_pred = np.zeros((row_follow, shape_1))
		y_final_val_pred[train_val_idx[i][1]] = y_val_pred
		if X_follow is not None:
			y_final_follow_pred += y_follow_pred
	if X_follow is not None:
		y_final_follow_pred /= cv

	return y_final_val_pred, y_final_follow_pred


def cross_validate_transform(model, X, y, X_follow=None, cv=5, predict_method='predict_proba',
                             n_jobs=None, verbose=0, pre_dispatch='2*n_jobs', parallel=True,
                             task='classification', random_state=None, **kwargs):
	if parallel:
		return parallel_cross_validate_transform(model=model, X=X, y=y, X_follow=X_follow, cv=cv,
		                                         predict_method=predict_method, n_jobs=n_jobs,
		                                         verbose=verbose, pre_dispatch=pre_dispatch, task=task,
		                                         random_state=random_state, **kwargs)
	return seq_cross_validate_transform(model=model, X=X, y=y, X_follow=X_follow, cv=cv,
	                                    predict_method=predict_method, task=task, random_state=random_state, **kwargs)


class CacheCrossValidator(object):
	def __init__(self, fold=3):
		self.fold = fold
		self.max_cv_sample_size = [0 for _ in range(self.fold)]
		self.cache_ = dict([(i, []) for i in range(self.fold)])

	def cross_validate_transform(self, model, X, y, X_follow=None, cv=5, predict_method='predict_proba',
	                             n_jobs=None, verbose=0, pre_dispatch='2*n_jobs', parallel=False,
	                             task='classification', random_state=None, **kwargs):
		if parallel:
			return self.parallel_cross_validate_transform(model=model, X=X, y=y, X_follow=X_follow, cv=cv,
			                                              predict_method=predict_method, n_jobs=n_jobs,
			                                              verbose=verbose, pre_dispatch=pre_dispatch, task=task,
			                                              random_state=random_state, **kwargs)
		return self.seq_cross_validate_transform(model=model, X=X, y=y, X_follow=X_follow, cv=cv,
		                                         predict_method=predict_method, task=task, random_state=random_state,
		                                         **kwargs)

	def parallel_cross_validate_transform(self, model, X, y, X_follow=None, cv=5, predict_method="predict_proba",
	                                      n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
	                                      task="classification", random_state=None, **kwargs):
		row_n = X.shape[0]
		y_final_val_pred = None
		y_final_follow_pred = None
		if len(y.shape) == 1 and task != "regression":
			logger.debug("start cv, StratifiedKFold")
			splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
		else:
			logger.debug("start cv, KFold")
			splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

		parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
		# start_time = time.time()
		train_val_idx = list(splitter.split(X, y))
		predictions = parallel(
			delayed(_fit_and_transform)(
				model=model, X=X, y=y, train_idx=train_idx, val_idx=val_idx,
				X_follow=X_follow, predict_method=predict_method)
			for train_idx, val_idx in train_val_idx)
		# print("CV {} cost {:.2f} s".format(cv, time.time() - start_time))
		for i, (y_val_pred, y_follow_pred) in enumerate(predictions):
			logger.debug("y_val_pred.shape={}, y_follow_pred.shape={}".format(
				y_val_pred.shape, y_follow_pred.shape if y_follow_pred is not None else None))
			if i == 0:
				shape_1 = y_val_pred.shape[1]
				y_final_val_pred = np.zeros((row_n, shape_1))
				if X_follow is not None:
					row_follow = X_follow.shape[0]
					y_final_follow_pred = np.zeros((row_follow, shape_1))
			y_final_val_pred[train_val_idx[i][1]] = y_val_pred
			if X_follow is not None:
				y_final_follow_pred += y_follow_pred
		if X_follow is not None:
			y_final_follow_pred /= cv

		return y_final_val_pred, y_final_follow_pred

	@staticmethod
	def _init_cache_process(X, y, X_follow, redis_address, splitter, return_dict, return_list):
		try:
			ray.init(redis_address=redis_address)
			train_val_idx = list(splitter.split(X, y))
			for fold_idx, (train_idx, val_idx) in enumerate(train_val_idx):
				X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]
				return_list.append(len(train_idx) + len(val_idx))
				X_train_or_id = ray.put(X_train)
				y_train_or_id = ray.put(y_train)
				X_follows_or_id = ray.put((X_val, X_follow))
				return_dict[fold_idx] = [X_train_or_id, y_train_or_id, X_follows_or_id]
				if isinstance(X_train_or_id, ray.ObjectID):
					print("[{}] INIT CACHE!!!!!!".format(fold_idx))
		except Exception as e:
			print(e)
		finally:
			pass

	def init_cache(self, X, y, X_follow=None, cv=3, task=None, random_state=None, redis_address=None):
		assert random_state is not None, "for initialize cache, you should assign exact random state"
		if len(y.shape) == 1 and task != "regression":
			logger.debug("start cv, StratifiedKFold")
			splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
		else:
			logger.debug("start cv, KFold")
			splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

		import multiprocessing
		mgr = multiprocessing.Manager()
		return_dict = mgr.dict()
		return_list = mgr.list()
		p = multiprocessing.Process(target=self._init_cache_process,
		                            args=(X, y, X_follow, redis_address, splitter, return_dict, return_list)
		                            )
		p.start()
		p.join(8)

		# if p.is_alive():
		# 	p.terminate()
		# 	kill_tree(p.pid)

		for i in range(cv):
			self.cache_[i] = return_dict[i]
			self.max_cv_sample_size[i] = return_list[i]
		# print("AFTER UPDATE: {}, {}".format(self.cache_, self.max_cv_sample_size))

	def _fit_and_transform_v2(self, fold_idx, model, X, y, train_idx, val_idx, X_follow=None,
	                          predict_method="predict_proba", distribute=False, **kwargs):
		"""
		使用 fit predict 接口，方便并行化，无需先 fit 然后再 predict

		:param model:
		:param X:
		:param y:
		:param train_idx:
		:param val_idx:
		:param X_follow:
		:param predict_method:
		:return:
		"""
		clf = copy.deepcopy(model)
		X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]

		X_train_or_id = X_train
		y_train_or_id = y_train
		X_follows_or_id = (X_val, X_follow)
		logger.debug("len(train_idx) + len(val_idx) = {},"
		             " max_cv_sample_size[{}] = {}".format(
			len(train_idx) + len(val_idx), fold_idx, self.max_cv_sample_size[fold_idx]))

		# if len(train_idx) + len(val_idx) > self.max_cv_sample_size[fold_idx]:
		# 	self.max_cv_sample_size[fold_idx] = len(train_idx) + len(val_idx)
		# 	X_train_or_id = ray.put(X_train)
		# 	y_train_or_id = ray.put(y_train)
		# 	X_follows_or_id = ray.put((X_val, X_follow))
		# 	self.cache_[fold_idx] = [X_train_or_id, y_train_or_id, X_follows_or_id]
		# 	if isinstance(X_train_or_id, ray.ObjectID):
		# 		logging.info("[{}] INIT CACHE!!!!!!".format(fold_idx))
		if len(train_idx) + len(val_idx) == self.max_cv_sample_size[fold_idx]:
			try:
				assert len(self.cache_[fold_idx]) == 3, "self.cache_ length = {}," \
				                                        " invalid".format(len(self.cache_[fold_idx]))
				X_train_or_id = self.cache_[fold_idx][0]
				y_train_or_id = self.cache_[fold_idx][1]
				X_follows_or_id = self.cache_[fold_idx][2]
			except Exception as e:
				logger.warning(e)
			finally:
				pass
			if isinstance(X_train_or_id, ray.ObjectID):
				logging.info("[{}] USING CACHE!!!!!!".format(fold_idx))
				try:
					ha = ray.get(X_train_or_id)
					print("ha = {}".format(ha.shape))
					ga = ray.get(y_train_or_id)
					print("ga = {}".format(ga.shape))
					ta = ray.get(X_follows_or_id)
					print("ta = {}".format(ta))
				except Exception as e:
					print(e)
					print(traceback.format_exc())
				finally:
					print("WHY NOT PRINT?")
			print("NOT PRINT?")
		y_val_pred, y_follow_pred = _fit_predict_proxy(estimator=clf, X=X_train_or_id, y=y_train_or_id,
		                                               X_follows=X_follows_or_id, predict_method=predict_method,
		                                               distribute=distribute, **kwargs)

		y_val_pred, y_follow_pred = _regularize_output(y_val_pred), _regularize_output(y_follow_pred)

		return y_val_pred, y_follow_pred

	def seq_cross_validate_transform(self, model, X, y, X_follow=None, cv=5, predict_method="predict_proba",
	                                 task="classification", random_state=None, **kwargs):
		row_n = X.shape[0]
		y_final_val_pred = None
		y_final_follow_pred = None
		if len(y.shape) == 1 and task != "regression":
			logger.debug("start cv, StratifiedKFold")
			splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
		else:
			logger.debug("start cv, KFold")
			splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

		train_val_idx = list(splitter.split(X, y))

		distribute = 0
		if 'distribute' in kwargs:
			distribute = kwargs['distribute']
			kwargs.pop('distribute')

		predictions = []
		for i, (train_idx, val_idx) in enumerate(train_val_idx):
			predictions.append(self._fit_and_transform_v2(
				fold_idx=i, model=model, X=X, y=y, train_idx=train_idx, val_idx=val_idx,
				X_follow=X_follow, predict_method=predict_method, distribute=distribute, **kwargs))

		for i, (y_val_pred, y_follow_pred) in enumerate(predictions):
			logger.debug("y_val_pred.shape={}, y_follow_pred.shape={}".format(
				y_val_pred.shape, y_follow_pred.shape if y_follow_pred is not None else None))
			if i == 0:
				shape_1 = y_val_pred.shape[1]
				y_final_val_pred = np.zeros((row_n, shape_1))
				if X_follow is not None:
					row_follow = X_follow.shape[0]
					y_final_follow_pred = np.zeros((row_follow, shape_1))
			y_final_val_pred[train_val_idx[i][1]] = y_val_pred
			if X_follow is not None:
				y_final_follow_pred += y_follow_pred
		if X_follow is not None:
			y_final_follow_pred /= cv

		return y_final_val_pred, y_final_follow_pred


if __name__ == "__main__":
	from sklearn.model_selection import cross_val_score
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.datasets import load_iris

	X, y = load_iris(return_X_y=True)
	a = RandomForestClassifier(random_state=0)
	print('a', cross_val_score(a, X, y))
	print('b', cross_validate_score(a, X, y, cv=3))
