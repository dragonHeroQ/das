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
from das.ArchitectureSearch.Evaluator.DisDeepArchiEvaluator import DisDeepArchiEvaluator
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool

logger = logging.getLogger(das.logger_name)


class DisEvaluatorProxy(BaseEvaluator):

	def __init__(self,
	             num_evaluators=2,
	             max_task_at_single_evaluator=1,
	             max_task_at_single_node=2,
	             evaluator_class=None,
	             evaluator_params=None,
	             n_folds=3,
	             evaluation_rule=None,
	             **kwargs):
		super(DisEvaluatorProxy, self).__init__(validation_strategy='cv', validation_strategy_args=n_folds,
		                                        evaluation_rule=evaluation_rule, **kwargs)
		self.n_folds = n_folds
		self.evaluator_pool = []
		self.evaluator_class = evaluator_class or DisDeepArchiEvaluator
		self.evaluator_params = evaluator_params
		self.num_evaluators = num_evaluators
		self.max_task_at_single_evaluator = max_task_at_single_evaluator
		self.max_task_at_single_node = max_task_at_single_node
		for i in range(self.num_evaluators):
			evaltor = self.evaluator_class.remote(**self.evaluator_params)
			self.evaluator_pool.append(evaltor)
		# how many tasks are assigned to this evaluator, max: self.max_task_at_single_evaluator
		self.evaluator_status = [0 for _ in range(self.num_evaluators)]
		self.evaluator_to_node = {}
		node_ip_futures = [x.get_local_ip.remote() for x in self.evaluator_pool]
		for i in range(self.num_evaluators):
			self.evaluator_to_node[i] = ray.get(node_ip_futures[i])
		self.num_nodes = len(set(self.evaluator_to_node.values()))
		# how many tasks running in this node
		self.node_status = dict([(node, 0) for node in self.evaluator_to_node.values()])
		self.max_tasks_once = self.num_evaluators * self.max_task_at_single_evaluator

	def select_an_evaluator(self):
		for i in range(len(self.evaluator_pool)):
			if self.evaluator_status[i] < self.max_task_at_single_evaluator:
				node = self.evaluator_to_node[i]
				if self.node_status[node] < self.max_task_at_single_node:
					self.evaluator_status[i] += 1
					self.node_status[node] += 1
					return i, self.evaluator_pool[i]
		return None, None

	def seq_evaluate(self, learning_tools, X, y,
	                 run_time_limit=240.0, random_state=None, **kwargs):
		final_results = []
		for i in range(len(learning_tools)):
			final_results.append(learning_tools[i].evaluate(learning_tool=learning_tools[i],
			                                                X=X, y=y, run_time_limit=run_time_limit,
			                                                random_state=random_state, **kwargs))
		return final_results

	def evaluate(self, learning_tools, X, y, run_time_limit=240.0,
	             random_state=None, parallel=True, total_time=240.0, **kwargs):
		if not isinstance(learning_tools, list):
			learning_tools = [learning_tools]
		if not parallel:
			return self.seq_evaluate(learning_tools=learning_tools, X=X, y=y,
			                         run_time_limit=run_time_limit, random_state=random_state, **kwargs)
		kwargs['bla'] = 'bla'
		time_ref = time.time()
		nlt = len(learning_tools)
		finished = [0 for _ in range(nlt)]
		final_results = [None for _ in range(nlt)]
		for i, lt in enumerate(learning_tools):
			if not isinstance(lt, BaseLearningTool):
				finished[i] = 1
				assert isinstance(lt, dict), ("learning_tools[{}] = {} is neither reward"
				                              " dict nor BaseLearningTool object".format(i, lt))
				final_results[i] = lt
		stage = 0
		while sum(finished) < nlt:  # there are some learning tools that haven't finished evaluation
			if time.time() - time_ref >= total_time - 0.1:
				break
			print("Stage {}".format(stage))
			result_futures = []
			result_futures_idx = []
			result_futures_evaltor_node = []
			num_tasks = 0
			for i, fin in enumerate(finished):
				if fin == 0 and num_tasks < self.max_tasks_once:  # haven't finished
					eval_idx, evaluator = self.select_an_evaluator()
					if eval_idx is None or evaluator is None:  # if there is no appropriate evaluator to choose
						break
					result_futures_idx.append(i)
					node = self.evaluator_to_node[eval_idx]
					print("LearningTool {} Allocate to {}, node = {}".format(i, eval_idx, node))
					result_futures_evaltor_node.append((eval_idx, node))
					result_futures.append(
						evaluator.evaluate.remote(
							learning_tools[i], X, y, run_time_limit, random_state, **kwargs)
					)
					num_tasks += 1
			true_results = ray.get(result_futures)
			for i in range(num_tasks):
				idx = result_futures_idx[i]
				finished[idx] = 1
				final_results[idx] = true_results[i]
				self.evaluator_status[result_futures_evaltor_node[i][0]] -= 1
				self.node_status[result_futures_evaltor_node[i][1]] -= 1
			stage += 1
		return final_results


if __name__ == '__main__':
	ray.init()
	from benchmarks.data.letter.load_letter import load_letter
	from benchmarks.data.digits.load_digits import load_digits
	from benchmarks.data.dexter.load_dexter import load_dexter
	from benchmarks.data.yeast.load_yeast import load_yeast
	from benchmarks.data.adult.load_adult import load_adult

	# logger.setLevel('DEBUG')
	x_train, x_test, y_train, y_test = load_yeast()
	from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
	learning_tool = DeepArchiLearningTool(n_block=2, n_classes=10, evaluation_rule='accuracy_score')
	evaluator = DisEvaluatorProxy(num_evaluators=4,
	                              evaluator_class=DisDeepArchiEvaluator,
	                              evaluator_params={'n_folds': 3, 'evaluation_rule': 'accuracy_score'},
	                              max_task_at_single_evaluator=1,
	                              max_task_at_single_node=4,
	                              n_folds=3, evaluation_rule='accuracy_score')
	learning_tools = [learning_tool.create_learning_tool(**{'b1_algo': 'ExtraTreesClassifier', 'b1_num': i,
	                                                        'b2_algo': 'RandomForestClassifier', 'b2_num': j})
	                  for i in range(1, 5) for j in range(1, 5)]
	print(len(learning_tools))
	start_time = time.time()
	final_results = evaluator.evaluate(learning_tools, X=x_train, y=y_train, parallel=True)
	print(final_results)
	print("Time Cost: {}".format(time.time() - start_time))
