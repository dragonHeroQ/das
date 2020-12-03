import copy
import random
import string
from math import log, pow
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace
random.seed(0)


def convert_category_to_int(X, ind=None):
	ind_flag = True

	if ind is None:
		ind_flag = False

	r, c = X.shape

	if not ind_flag:

		for i in range(c):

			category = set(X[:, i])
			category_dict = dict(zip(category, [i for i in range(len(category))]))

			for j in range(r):
				X[j, i] = category_dict[X[j, i]]

	else:

		for i in ind:

			category = set(X[:, i])

			category_dict = dict(zip(category, [i for i in range(len(category))]))

			for j in range(r):
				X[j, i] = category_dict[X[j, i]]
	return X


def convert_category_to_int_by_parameterspace(X, parameter_space):
	res = []
	i = -1
	for ps in parameter_space.get_space():
		i = i + 1
		# print(type(ps), isinstance(ps, CategorySpace))
		if not isinstance(ps, CategorySpace):
			res.append(X[i])
			continue

		else:
			choice_space = ps.get_choice_space()
			# category_dict = dict(zip(choice_space, [i for i in range(len(choice_space))]))
			# print(category_dict)
			# res.append(category_dict[X[i]])
			for num in range(len(choice_space)):
				if choice_space[num] == X[i]:
					res.append(num)

	return res


def get_category_data(X, parameter_space):
	"""

	:param X: data
	:param parameter_space:
	:return: category data and its parameter space and other type data
	"""

	res = []
	other_data = []
	res_space = []
	i = -1
	for ps in parameter_space.get_space():
		i = i + 1
		# print(i, X[i])
		if not isinstance(ps, CategorySpace):
			other_data.append(X[i])

		else:
			res.append(X[i])
			res_space.append(ps)
	return res, res_space, other_data


def category_to_onehot(X, encoder):
	return encoder.transform(X)


class AbstractParameterSpace(object):
	"""
	base class
	"""

	def __init__(self, name=None, min_val=None, max_val=None, default=None):
		self._default = default
		self._name = name
		self._min_val = min_val
		self._max_val = max_val

	def get_default(self):
		return self._default

	def set_default(self, new_default):
		self._default = new_default

	def get_name(self):
		return self._name

	def set_name(self, new_name):
		self._name = new_name

	def get_min_val(self):
		return self._min_val

	def set_min_val(self, new_min_val):
		self._min_val = new_min_val

	def get_max_val(self):
		return self._max_val

	def set_max_val(self, new_max_val):
		self._max_val = new_max_val

	def get_random_val(self):
		raise NotImplementedError

	def __repr__(self):
		return self.__class__.__name__


class UniformFloatSpace(AbstractParameterSpace):
	"""
	float parameter space and the distribution of this space is uniform
	"""

	def __init__(self, name, min_val, max_val, default=None):
		super(UniformFloatSpace, self).__init__(name, min_val, max_val, default)
		self._name = name
		self._min_val = min_val
		self._max_val = max_val

		if default:
			self._default = default
		else:
			self._default = self._min_val

	def get_random_val(self):
		return random.uniform(self._min_val, self._max_val)


class UniformIntSpace(AbstractParameterSpace):
	"""
	Integer parameter space and the distribution of this space is uniform
	"""

	def __init__(self, name, min_val, max_val, default=None):
		super(UniformIntSpace, self).__init__(name, min_val, max_val, default)

		if default:
			self._default = default
		else:
			self._default = self._min_val

	def get_random_val(self):
		return random.randrange(self._min_val, self._max_val+1)


class LogFloatSpace(AbstractParameterSpace):
	"""
	log float parameter space
	"""

	def __init__(self, name, min_val, max_val, base=10, default=None):
		super(LogFloatSpace, self).__init__(name, min_val, max_val, default)
		self._base = base
		if default:
			self._default = default
		else:
			self._default = self._min_val

	def get_random_val(self):
		a = log(self._min_val, self._base)
		b = log(self._max_val, self._base)
		sel = random.uniform(a, b)
		return pow(self._base, sel)


class LogIntSpace(AbstractParameterSpace):
	"""
	log integer parameter space
	"""

	def __init__(self, name, min_val, max_val, base=10, default=None):
		super(LogIntSpace, self).__init__(name, min_val, max_val, default)
		self._base = base

		if default:
			self._default = default
		else:
			self._default = self._min_val

	def get_random_val(self):
		a = log(self._min_val, self._base)
		b = log(self._max_val, self._base)
		sel = random.uniform(a, b)
		return int(pow(self._base, round(sel)))


class RangeSpace(AbstractParameterSpace):
	"""
	range space
	"""

	def __init__(self, name, min_val, max_val, step, default=None):
		super(RangeSpace, self).__init__(name, min_val, max_val, default)
		self._step = step

		if default:
			self._default = default
		else:
			self._default = self._min_val

	def get_random_val(self):
		raise NotImplementedError


class CategorySpace(AbstractParameterSpace):
	"""
	category space
	"""

	def __init__(self, name, choice_space, default=None):
		super(CategorySpace, self).__init__(name, default=default)

		self._choice_space = choice_space
		self._space_size = len(choice_space)

		if default:
			self._default = default
		else:
			self._default = self.get_random_val()

	def get_choice_space(self):
		return self._choice_space

	def set_choice_space(self, new_choice_space):
		self._choice_space = new_choice_space

	def get_random_val(self):
		if self._space_size == 1:
			return self._choice_space[0]
		else:
			sel = random.randrange(0, self._space_size)
		return self._choice_space[sel]


class ParameterSpace(object):

	def __init__(self):
		self._space = []
		self._constraint = []
		self._relations = []

	def merge(self, spaces):

		for space in spaces:
			self._space.append(space)

	def remove_hyperparameter(self, name=None):
		if name is None:
			pass
		for idx in range(len(self._space) - 1, -1, -1):
			if self._space[idx].get_name() == name:
				del self._space[idx]

	def add_parameter_relation(self, relation):
		self._relations.extend(relation)

	def check_parameter(self, params):

		for rel in self._relations:
			if not rel.judge(params):
				return False
		return True

	def get_parameter_relation(self):
		return copy.deepcopy(self._relations)

	def get_space(self):
		return copy.deepcopy(self._space)

	def get_space_names(self):
		return list(map(lambda x: (x.get_name(), str(x)), self._space))

	def show_space_names(self):
		print("Configuration Space: ")
		spaces = self.get_space_names()
		for space in spaces:
			print(" {}: {}".format(space[0], space[1]))

	def get_default_config(self):
		res = {}
		for space in self._space:
			res[space.get_name()] = space.get_default()
		return res

	def get_onehot_encoder(self):
		return self._make_onehot_encoder()

	def get_random_config(self):
		while True:
			res = {}
			for space in self._space:
				res[space.get_name()] = space.get_random_val()
			if self.check_parameter(res):
				break
		return res

	def _make_onehot_encoder(self):

		cat_space = []
		for ps in self._space:
			if not isinstance(ps, CategorySpace):
				continue
			else:
				cat_space.append(ps)

		max_size = 0
		for cs in cat_space:
			if len(cs.get_choice_space()) > max_size:
				max_size = len(cs.get_choice_space())

		data = []
		for i in range(max_size):
			temp_data = []
			for j in range(len(cat_space)):
				cs = cat_space[j].get_choice_space()
				if len(cs) > i:
					temp_data.append(cs[i])
				else:
					temp_data.append(cs[-1])
			data.append(temp_data)

		data = convert_category_to_int(np.array(data))
		enc = OneHotEncoder()
		enc.fit(data)

		return enc

	def to_config_space(self):
		cs = ParamSpace2ConfigSpace(self)
		return cs


class ConfigTransformer(object):
	"""
	Transform the config (all str) to config that may contains object, functions, et.al

	Parameters
	----------
	nick2ground: nickname (str) to true object
	ground2nick: true object to nickname (str)
	"""

	def __init__(self, nick2ground=None, ground2nick=None):
		self.nick2ground = nick2ground
		self.ground2nick = ground2nick
		if self.nick2ground is None:
			self.nick2ground = {}
		if self.ground2nick is None:
			self.ground2nick = {}

	def _get_strground2nick(self):
		strground2nick = {}
		for key in self.ground2nick:
			strground2nick[str(key)] = self.ground2nick[key]
		return strground2nick

	def __call__(self, config=None, nick2ground=True):
		# print("Transform: config = {}, self.nick2ground = {}".format(config, self.nick2ground))
		if config is None:
			return None
		if nick2ground:
			if len(self.nick2ground) == 0:
				return config
			for key in config.keys():
				if config[key] in self.nick2ground:
					config[key] = self.nick2ground[config[key]]
			return config
		else:
			if len(self.ground2nick) == 0:
				return config
			strground2nick = self._get_strground2nick()
			for key in config.keys():
				# print("{} in self.ground2nick={} ? {}".format(config[key], self.ground2nick, str(config[key]) \
				#  in list(map(lambda x: str(x), self.ground2nick))))
				if str(config[key]) in strground2nick:
					config[key] = strground2nick[str(config[key])]
			return config

	def __repr__(self):
		return "%s (%s), %s (%s)" % ("nick2ground", self.nick2ground, "ground2nick", self.ground2nick)


class DASConfigurationSpace(ConfigurationSpace):
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
	cs = DASConfigurationSpace()
	nick2ground = {}
	ground2nick = {}
	for space in das_parameter_space.get_space():
		if isinstance(space, CategorySpace):
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
		elif isinstance(space, UniformIntSpace):
			assert space.get_min_val() < space.get_max_val(), ("UniformIntegerSpace, min_val={}"
			                                                   " should smaller than max_val={}".format(
				space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformIntegerHyperparameter(space.get_name(), lower=space.get_min_val(),
				                                 upper=space.get_max_val()))
		elif isinstance(space, UniformFloatSpace):
			assert space.get_min_val() <= space.get_max_val(), ("UniformFloatSpace, min_val={}"
			                                                    " should smaller than max_val={}".format(
				space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformFloatHyperparameter(space.get_name(), lower=space.get_min_val(), upper=space.get_max_val()))
		elif isinstance(space, LogIntSpace):
			assert space.get_min_val() < space.get_max_val(), ("LogIntegerSpace, min_val={}"
			                                                   " should smaller than max_val={}".format(
				space.get_min_val(), space.get_max_val()))
			cs.add_hyperparameter(
				CSH.UniformIntegerHyperparameter(space.get_name(),
				                                 lower=space.get_min_val(), upper=space.get_max_val(), log=True))
		elif isinstance(space, LogFloatSpace):
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


if __name__ == "__main__":
	print(__name__)

	import numpy as np
	from sklearn.preprocessing import OneHotEncoder

	a = [["cat", 1, 2], ["dog", 2, 3], ["pig", 3, 4]]
	a = np.array(a)
	print(a)
	# print(convert_catogory_to_int(a))

	f_space = CategorySpace("f", choice_space=["cat", "dog", "pig"], default="cat")
	s_space = UniformIntSpace("s", min_val=1, max_val=3, default=1)
	t_space = UniformIntSpace("t", min_val=2, max_val=4, default=2)
	ps = ParameterSpace()
	ps.merge([f_space, s_space, t_space])

	for i in ps.get_space():
		print(i.get_name(), i.get_min_val(), i.get_max_val(), i.get_default(), i.get_random_val())
