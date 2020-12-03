

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
