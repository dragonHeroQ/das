import copy
from das.ParameterSpace import ParamSpace2ConfigSpace


class BaseLearningTool(object):

	def __init__(self, n_classes=None, evaluation_rule=None, cross_validator=None):
		self.parameter_space = None
		self.learning_estimator = None
		self.n_classes_ = n_classes or 1
		self.evaluation_rule = evaluation_rule
		self.cross_validator = cross_validator

	def set_parameter_space(self, ps=None):
		raise NotImplementedError

	def get_parameter_space(self):
		if self.parameter_space is None:
			self.set_parameter_space()
		return copy.deepcopy(self.parameter_space)

	def get_config_space(self):
		return ParamSpace2ConfigSpace(self.parameter_space)

	def set_classes(self, n_class):
		self.n_classes_ = n_class

	def set_evaluation_rule(self, evaluation_rule=None):
		self.evaluation_rule = evaluation_rule

	def create_learning_tool(self, **hyper_params):
		raise NotImplementedError

