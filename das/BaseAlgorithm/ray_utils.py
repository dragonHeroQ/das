import ray
import das
import copy
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from das.crossvalidate import fit_and_transform_v2

logger = logging.getLogger(das.logger_name)


@ray.remote
def ray_fit_predict(algorithm, X, y, X_follows=None, predict_method="predict_proba"):
	algorithm.fit(X, y)
	return tuple(getattr(algorithm, predict_method)(X_follows[i])
	             if X_follows[i] is not None else None
	             for i in range(len(X_follows)))


@ray.remote
def block_fit_predict(block, X, y, X_follows=None, predict_method="predict_proba", distribute=False):
	return block.fit_predict(X, y, X_follows, predict_method, distribute)


@ray.remote
def block_fit_predict_kfold(block_model, Xid, y, Xcat, cv, predict_method, task, random_state, distribute):
	print("block_model.fit_predict_kfold_ray ...")
	return block_model.fit_predict_kfold_ray(Xid=Xid, y=y, Xcat=Xcat, cv=cv, predict_method=predict_method,
	                                         task=task, random_state=random_state, distribute=distribute)


@ray.remote
def unit_fit_predict_kfold(unit_model, Xid, y, Xcat, cv, predict_method, task, random_state, distribute):
	row_n = y.shape[0]
	y_final_val_pred = None
	if (len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)) and task != "regression":
		# logger.debug("start cv, StratifiedKFold")
		splitter = StratifiedKFold(n_splits=cv, random_state=random_state)
	else:
		# logger.debug("start cv, KFold")
		splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

	X = np.hstack((Xid, Xcat))
	# logger.debug("Unit FitPredict KFold: Concatenated X shape = {}".format(X.shape))
	train_val_idx = list(splitter.split(X, y))

	predictions = []
	for train_idx, val_idx in train_val_idx:
		predictions.append(fit_and_transform_v2(
			model=unit_model, X=X, y=y, train_idx=train_idx, val_idx=val_idx,
			X_follow=None, predict_method=predict_method, distribute=distribute))

	for i, (y_val_pred, _) in enumerate(predictions):
		if y_val_pred is None:  # if any exception caused None for y_val_pred, we directly return None
			return None
		if i == 0:
			shape_1 = y_val_pred.shape[1]
			y_final_val_pred = np.zeros((row_n, shape_1))

		y_final_val_pred[train_val_idx[i][1]] = y_val_pred

	return y_final_val_pred


@ray.remote
def get_local_ip():
	return ray.services.get_node_ip_address()



