import das
import ray
import time
import logging
import traceback
import numpy as np
import multiprocessing
from das.util.common_utils import kill_tree
from das.performance_evaluation import score_to_loss, loss_to_score
from das.ArchitectureSearch.Evaluator.BaseEvaluator import BaseEvaluator

logger = logging.getLogger(das.logger_name)


class DeepArchiEvaluator(BaseEvaluator):

	def __init__(self,
	             n_folds=3,
	             evaluation_rule=None,
	             redis_address=None,
	             **kwargs):
		super(DeepArchiEvaluator, self).__init__(validation_strategy='cv', validation_strategy_args=n_folds,
		                                         evaluation_rule=evaluation_rule, **kwargs)
		self.n_folds = n_folds
		self.redis_address = redis_address
		self.Xid = None

	@staticmethod
	def _fit_and_score(learning_estimator, X, y,
	                   return_dict=None, random_state=None, redis_address=None, **kwargs):
		import ray
		try:
			if 'distribute' in kwargs and kwargs['distribute'] > 0:
				# print("_fit_and_score: distribute = {}".format(kwargs['distribute']))
				if redis_address is None:
					ray.init()
				else:
					ray.init(redis_address=redis_address)
				logger.info("Ray Initialized")
		except Exception as e:
			logger.info(traceback.format_exc())
			logger.info("RESTARTING RAY CLUSTER ...")
			import os
			os.system("sh /home/experiment/huqiu/das/benchmarks/restart.sh")
			if redis_address is None:
				ray.init()
			else:
				ray.init(redis_address=redis_address)
			logger.info("After Exception, Ray Initialized")
		try:
			if 'distribute' in kwargs and kwargs['distribute'] > 0:
				if not isinstance(X, ray.ObjectID):
					logger.warning("fit_ray requires X=ObjectID but X={}".format(type(X)))
					Xid = ray.put(X)
					return_dict['Xid'] = Xid
					X = Xid
					logger.info("Put down X, got Xid = {}".format(Xid))
				learning_estimator.fit_ray(X, y, random_state=random_state, **kwargs)
			else:
				learning_estimator.fit(X, y, random_state=random_state, **kwargs)
		except Exception as e:
			print("[DeepArchiEvaluator: 61]", traceback.format_exc())
			print("hasattr(learning_estimator, opt_train_metric) = {}".format(
				hasattr(learning_estimator, 'opt_train_metric')))
			print("hasattr(learning_estimator, best_num_layers) = {}".format(
				hasattr(learning_estimator, 'best_num_layers')))
			return_dict['exception'] = str(e)
		finally:
			return_dict['opt_metric'] = learning_estimator.opt_train_metric
			return_dict['best_num_layers'] = learning_estimator.best_num_layers
			if redis_address is None:
				ray.shutdown()

	def evaluate(self, learning_tool, X, y, run_time_limit=240.0, random_state=None, **kwargs):
		learning_estimator = learning_tool.learning_estimator
		learning_estimator.n_folds = self.n_folds
		loss = self.worst_loss
		best_num_layers = 0
		start_time = time.time()
		error_message = None

		if self.Xid is None and 'distribute' in kwargs and kwargs['distribute'] > 0:
			# self.put_X_update_Xid(X)
			X = self.Xid or X
		elif self.Xid is None and ('distribute' not in kwargs or kwargs['distribute'] <= 0):
			pass
		elif self.Xid is not None:
			X = self.Xid
			logger.info("Using off-the-shelf Xid = {}".format(X))

		try:
			mgr = multiprocessing.Manager()
			return_dict = mgr.dict()
			p = multiprocessing.Process(target=self._fit_and_score,
			                            args=(learning_estimator, X, y, return_dict, random_state, self.redis_address),
			                            kwargs=kwargs)
			p.start()
			logger.info("RunTimeLimit = {}".format(run_time_limit))
			p.join(run_time_limit)

			if p.is_alive():
				print("Additional Terminating...")
				p.terminate()
				kill_tree(p.pid)

			if 'opt_metric' in return_dict and return_dict['opt_metric'] is not None:
				val_score = return_dict['opt_metric']
				best_num_layers = return_dict['best_num_layers']
				if 'exception' in return_dict:
					error_message = return_dict['exception']
			else:
				# print("555~ TimeOut, opt_train_metric = ", learning_estimator.opt_train_metric) = None
				error_message = 'May be TimeOut'
				if 'exception' in return_dict:
					error_message = return_dict['exception']
				val_score = loss_to_score(self.evaluation_rule, self.worst_loss)
				best_num_layers = 0

			loss = score_to_loss(self.evaluation_rule, val_score)
			if 'Xid' in return_dict and return_dict['Xid'] is not None:
				self.Xid = return_dict['Xid']
		except Exception as e:
			logger.warning("Exceptions Occurred")
			error_message = str(e)
			print(traceback.format_exc())
		finally:
			time_cost = time.time()-start_time
			reward = {'loss': loss,
			          'val_{}'.format(self.evaluation_rule):
				          loss_to_score(rule=self.evaluation_rule, loss=loss),
			          'best_nLayer': best_num_layers,
			          'time_cost': time_cost,
			          'exception': error_message}
			self.update_best_val_score(reward['val_{}'.format(self.evaluation_rule)])
			# learning_tool.learning_estimator = None  # clear learning estimator to save memory
			self.record_learning_curve(reward=reward, learning_tool=learning_tool,
			                           model_params=learning_estimator.get_params())
			return reward

