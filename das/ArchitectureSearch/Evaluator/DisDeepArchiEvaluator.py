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
from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool

logger = logging.getLogger(das.logger_name)


@ray.remote
class DisDeepArchiEvaluator(BaseEvaluator):

	def __init__(self,
	             n_folds=3,
	             evaluation_rule=None):
		super().__init__(validation_strategy='cv', validation_strategy_args=n_folds,
		                 evaluation_rule=evaluation_rule)
		self.n_folds = n_folds
		self.validation_strategy = 'cv'
		self.validation_strategy_args = n_folds
		self.evaluation_rule = evaluation_rule

	@staticmethod
	def _fit_and_score(learning_estimator, X, y,
	                   return_dict=None, random_state=None, **kwargs):
		try:
			learning_estimator.fit(X, y, random_state=random_state, **kwargs)
		except Exception as e:
			return_dict['exception'] = str(e)
		finally:
			return_dict['opt_metric'] = learning_estimator.opt_train_metric
			return_dict['best_num_layers'] = learning_estimator.best_num_layers

	def get_local_ip(self):
		return ray.services.get_node_ip_address()

	def evaluate(self, learning_tool, X, y, run_time_limit=240.0,
	             random_state=None, bla='', confidence_screening=True, debug=False):
		# fake
		# time.sleep(5)
		# return {'val_score': 0.3, 'loss': 0.5}
		learning_estimator = learning_tool.learning_estimator
		learning_estimator.n_folds = self.n_folds
		loss = self.worst_loss
		best_num_layers = 0
		start_time = time.time()
		error_message = None
		try:
			mgr = multiprocessing.Manager()
			return_dict = mgr.dict()
			p = multiprocessing.Process(target=self._fit_and_score,
			                            args=(learning_estimator, X, y, return_dict, random_state),
			                            kwargs={'confidence_screening': confidence_screening})
			p.start()
			p.join(run_time_limit)

			if p.is_alive():
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

