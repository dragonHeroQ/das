from das.ParameterSpace import ParameterSpace
from das.util.decorators import check_parameter_space_not_none


class BaseOptimizer(object):

	def __init__(self, parameter_space: ParameterSpace=None, **kwargs):
		self.parameter_space = parameter_space

		for key in kwargs:
			setattr(self, key, kwargs[key])

	@check_parameter_space_not_none
	def get_default_config(self):
		return self.parameter_space.get_default_config()

	@check_parameter_space_not_none
	def get_random_config(self):
		return self.parameter_space.get_random_config()

	def get_next_config(self):
		raise NotImplementedError

	def get_debug_config(self):
		return {'b1_algo': 'LGBClassifier', 'b1_num': 4, 'b2_algo': 'MLPClassifier', 'b2_num': 1}

	def new_result(self, config, reward, other_infos=None, update_model=True):
		pass


