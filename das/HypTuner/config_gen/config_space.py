import random
import string
import das.ParameterSpace as PS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace
from das.HypTuner.config_gen.config_transformer import ConfigTransformer


class PasaConfigurationSpace(ConfigurationSpace):
	"""
	PasaConfigurationSpace to perform hyper-parameter optimization.
	Add config transformer from das.ParamSpace to ConfigSpace.ConfigurationSpace
	"""
	def set_config_transformer(self, config_transformer):
		self.config_transformer = config_transformer

	def get_config_transformer(self):
		if hasattr(self, "config_transformer"):
			return self.config_transformer
		else:
			return ConfigTransformer()


def ParamSpace2ConfigSpace(das_parameter_space=None):
	"""
	Convert das.ParamSpace to ConfigSpace.ConfigurationSpace

	Parameters
	----------
	das_parameter_space
		das.ParamSpace instance
	Returns
	-------

	"""
	cs = PasaConfigurationSpace()
	nick2ground = {}
	ground2nick = {}
	for space in das_parameter_space.get_space():
		if isinstance(space, PS.CategorySpace):
			choice = space.get_choice_space()
			for i, choi in enumerate(choice):
				if not is_legal_config_item(choi):
					# print("{} is not a legal item!".format(choi))
					nickname = get_nickname(choi)
					# print("choi = {}, nickname = {}".format(choi, nickname))
					ground2nick[choi] = nickname
					# print(ground2nick)
					nick2ground[nickname] = choi
					choice[i] = nickname
			cs.add_hyperparameter(CSH.CategoricalHyperparameter(space.get_name(), choice))
		elif isinstance(space, PS.UniformIntSpace):
			assert space.get_min_val() < space.get_max_val(), ("UniformIntegerSpace, min_val={}"
			                                                   " should smaller than max_val={}".format(
				                                               space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformIntegerHyperparameter(space.get_name(), lower=space.get_min_val(), upper=space.get_max_val()))
		elif isinstance(space, PS.UniformFloatSpace):
			assert space.get_min_val() <= space.get_max_val(), ("UniformFloatSpace, min_val={}"
			                                                    " should smaller than max_val={}".format(
				                                                space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformFloatHyperparameter(space.get_name(), lower=space.get_min_val(), upper=space.get_max_val()))
		elif isinstance(space, PS.LogIntSpace):
			assert space.get_min_val() < space.get_max_val(), ("LogIntegerSpace, min_val={}"
			                                                   " should smaller than max_val={}".format(
				                                               space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformIntegerHyperparameter(space.get_name(),
				                                 lower=space.get_min_val(), upper=space.get_max_val(), log=True))
		elif isinstance(space, PS.LogFloatSpace):
			assert space.get_min_val() < space.get_max_val(), ("LogFloatSpace, min_val={}"
			                                                   " should smaller than max_val={}".format(
				                                               space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformFloatHyperparameter(space.get_name(),
				                               lower=space.get_min_val(), upper=space.get_max_val(), log=True))
		else:
			raise NotImplementedError("space type is not supported till now!")
	# print("nick2ground = {}".format(nick2ground))
	# print("ground2nick = {}".format(ground2nick))
	tsf = ConfigTransformer(nick2ground=nick2ground, ground2nick=ground2nick)
	cs.set_config_transformer(tsf)
	return cs


def is_legal_config_item(cfg):
	"""

	Parameters
	----------
	cfg

	Returns
	-------

	"""
	return isinstance(cfg, (float, int, str, bool, list, tuple, set, dict))


def get_nickname(cfg):
	"""
	Convert object to str, because ConfigSpace does not support object hyper-parameters.

	Parameters
	----------
	cfg

	Returns
	-------

	"""
	if isinstance(cfg, object):
		if hasattr(cfg, '__name__'):
			return cfg.__name__
		return cfg.__class__.__name__
	else:
		rand_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
		return rand_str


if __name__ == '__main__':
	from sklearn.feature_selection import chi2

	# print(type(ParamSpace2ConfigSpace))
	print(get_nickname(chi2))
	# print(type(chi2))
	from das.BaseAlgorithm.Classification.LogisticRegression import LogisticRegression
	print(get_nickname(LogisticRegression()))
