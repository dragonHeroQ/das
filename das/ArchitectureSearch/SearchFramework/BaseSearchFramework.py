import time
import copy
from das.performance_evaluation import initial_worst_loss, judge_rule
from das.ArchitectureSearch.SamplingStrategy import SamplingStrategy
from das.ArchitectureSearch.Optimizer.BaseOptimizer import BaseOptimizer
from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import BaseEvaluator
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool


class BaseSearchFramework(object):

	def __init__(self,
	             optimizer: BaseOptimizer=None,
	             evaluator: BaseEvaluator=None,
	             learning_tool: BaseLearningTool=None,
	             sampling_strategy: SamplingStrategy=None,
	             search_space=None,
	             total_budget=10,
	             budget_type="trial",
	             per_run_timelimit=240.0,
	             evaluation_rule=None,
	             cross_validator=None,
	             time_ref=None,
	             worst_loss=None,
	             task=None,
	             random_state=None,
	             **kwargs
	             ):
		self.optimizer = optimizer
		self.evaluator = evaluator
		self.learning_tool = learning_tool
		self.sampling_strategy = sampling_strategy
		self.search_space = search_space

		self.total_budget = total_budget
		self.budget_type = budget_type
		assert self.budget_type in ['trial', 'time'], "Unsupported budget type: {}".format(self.budget_type)
		self.per_run_timelimit = per_run_timelimit
		self.evaluation_rule = evaluation_rule
		assert self.evaluation_rule is not None, "Evaluation Rule should be provided!"
		self.learning_tool.set_evaluation_rule(self.evaluation_rule)
		self.cross_validator = cross_validator
		self.time_ref = time_ref or time.time()
		self.worst_loss = worst_loss or initial_worst_loss(rule=self.evaluation_rule)
		self.task = judge_rule(self.evaluation_rule)
		self.random_state = random_state

		for key in kwargs:
			setattr(self, key, kwargs[key])

		# running attrs
		self.learning_curve = {}   # {time: performance}

	def fit(self, X, y, **fit_params):
		raise NotImplementedError

	def refit(self, X, y, **refit_params):
		raise NotImplementedError

	def refit_transform(self, X, y, X_test, **refit_params):
		raise NotImplementedError

	def refit_and_score(self, X, y, X_test, y_test, **refit_params):
		raise NotImplementedError

	def get_learning_curve(self):
		return copy.deepcopy(self.learning_curve)

