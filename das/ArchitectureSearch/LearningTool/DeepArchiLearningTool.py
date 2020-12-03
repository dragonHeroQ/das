import das
import sys
import logging
import traceback
import numpy as np
from das.ParameterSpace import *
from das.util.hash_utils import get_hash
from das.performance_evaluation import judge_rule
from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier, VerticalBlockClassifier
from das.BaseAlgorithm.Classification.ArchiLayerClassifier import ArchiLayerClassifier
from das.BaseAlgorithm.Classification.DeepArchiClassifier import DeepArchiClassifier
from das.BaseAlgorithm.Regression.ArchiBlockRegressor import HorizontalBlockRegressor, VerticalBlockRegressor
from das.BaseAlgorithm.Regression.ArchiLayerRegressor import ArchiLayerRegressor
from das.BaseAlgorithm.Regression.DeepArchiRegressor import DeepArchiRegressor
from das.BaseAlgorithm.algorithm_space import (get_algorithm_class_by_key, get_all_classification_algorithm_keys,
                                               get_all_regression_algorithm_keys, get_algorithm_key_dict)
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool

logger = logging.getLogger(das.logger_name)
np.random.seed(0)


class DeepArchiLearningTool(BaseLearningTool):

	def __init__(self, n_block=2, n_classes=None, evaluation_rule=None, algo_space=None, **kwargs):
		super(DeepArchiLearningTool, self).__init__(n_classes=n_classes, evaluation_rule=evaluation_rule, **kwargs)
		self.n_block = n_block
		self.learning_estimator = None
		self.last_model_storage_size = None
		self.hyper_params = None
		self.candidate_mutation_ops = None
		self.hash2modelparams = {}
		self.algo_space = algo_space
		if self.algo_space is None:
			if self.is_classification:
				self.algo_space = get_all_classification_algorithm_keys()
			else:
				self.algo_space = get_all_regression_algorithm_keys()

	@property
	def is_classification(self):
		return judge_rule(self.evaluation_rule) == 'classification' and self.n_classes_ not in [None, 1]

	def set_parameter_space(self, ps=None):
		parameter_space = ParameterSpace()
		if ps is None:

			b1_num = UniformIntSpace(name="b1_num", min_val=1, max_val=5, default=2)
			b1_algo = CategorySpace(name="b1_algo", choice_space=self.algo_space)
			b2_num = UniformIntSpace(name="b2_num", min_val=1, max_val=5, default=2)
			b2_algo = CategorySpace(name="b2_algo", choice_space=self.algo_space)

			parameter_space.merge([b1_num, b1_algo, b2_num, b2_algo])
		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space

	def create_learning_tool(self, max_layer=0, early_stopping_rounds=1,
	                         n_folds=3, cross_validator=None, **hyper_params):
		self.hyper_params = hyper_params
		# Note: it's ok to let self.n_classes_ be None, Classifier themselves can infer n_classes in the runtime
		# assert self.n_classes_ is not None, "Please set n_classes_ before creating a learning tool."
		if self.is_classification:
			block_class = HorizontalBlockClassifier
			layer_class = ArchiLayerClassifier
			model_class = DeepArchiClassifier
		else:
			block_class = HorizontalBlockRegressor
			layer_class = ArchiLayerRegressor
			model_class = DeepArchiRegressor
		blocks = []
		for i in range(self.n_block):
			blocks.append(("block{}".format(i),
			               block_class(nc=hyper_params['b{}_num'.format(i+1)],
			                           model_class=get_algorithm_class_by_key(
                                          hyper_params['b{}_algo'.format(i+1)]),
			                           n_classes=self.n_classes_, e_id=i, random_state=i)))
		basic_layer = layer_class(nc=len(blocks),
		                          model=blocks,
		                          n_classes=self.n_classes_,
		                          e_id=0, random_state=0)
		if 'params' in hyper_params:
			try:
				basic_layer.set_params(**hyper_params['params'])
			except Exception as e:
				print(traceback.format_exc())
				print("hyper_params['params'] = ", hyper_params['params'])
				print("basic_layer = ", basic_layer.get_model_name(), basic_layer.get_params())
		cross_validator = self.cross_validator if cross_validator is None else cross_validator
		deep_model = model_class(base_layer=basic_layer,
		                         max_layer=max_layer,
		                         early_stopping_rounds=early_stopping_rounds,
		                         n_folds=n_folds,
		                         evaluation_rule=self.evaluation_rule,
		                         cross_validator=cross_validator,
		                         n_classes=self.n_classes_,
		                         e_id=0, random_state=0)
		self.last_model_storage_size = (sys.getsizeof(deep_model) / 1024.0)
		if self.last_model_storage_size > 5000:
			logger.warning("model size are larger than 5000 KB (5MB), please take attention!")

		# return deep_model
		return_learning_tool = DeepArchiLearningTool(n_block=self.n_block,
		                                             n_classes=self.n_classes_,
		                                             evaluation_rule=self.evaluation_rule)
		return_learning_tool.learning_estimator = deep_model
		return_learning_tool.last_model_storage_size = self.last_model_storage_size

		return return_learning_tool

	def encode_archi(self, archi_config, **kwargs):
		return self.encode_archi_v2(archi_config, **kwargs)

	def encode_archi_v1(self, archi_config, **kwargs):
		key_dict = get_algorithm_key_dict(self.algo_space)
		# encode_string = "{}#".format(archi_config['random_state'])
		encode_string = ""
		for i in range(2):
			b_algo = archi_config['b{}_algo'.format(i+1)]
			b_num = archi_config['b{}_num'.format(i+1)]
			encode_string += "H@{}@{}".format(b_num, key_dict[b_algo])
			if i < 1:
				encode_string += '#'

		return encode_string

	def encode_archi_v2(self, archi_config, learning_tool=None, params=None, **kwargs):

		key_dict = get_algorithm_key_dict(self.algo_space)

		if 'params' in archi_config:
			params_hash = get_hash(archi_config['params'])
			true_params = archi_config['params']
		elif learning_tool is not None:
			params_hash = get_hash(learning_tool.learning_estimator.get_params())
			true_params = learning_tool.learning_estimator.get_params()
		else:
			assert params is not None, "LearningTool and params should not be all None!"
			params_hash = get_hash(params)
			true_params = params
		self.hash2modelparams[params_hash] = true_params
		encode_string = "{}#".format(params_hash)

		# encode_string = ""
		for i in range(2):
			b_algo = archi_config['b{}_algo'.format(i+1)]
			b_num = archi_config['b{}_num'.format(i+1)]
			encode_string += "H@{}@{}".format(b_num, key_dict[b_algo])
			if i < 1:
				encode_string += '#'

		# print("when encode {}, self.hash2modelparams[{}] = {}".format(encode_string, params_hash, true_params))
		return encode_string

	def decode_archi(self, encoded_archi):
		return self.decode_archi_v2(encoded_archi)

	def decode_archi_v2(self, encoded_archi):
		archi_config = {}
		key_list = self.algo_space
		blocks = encoded_archi.split('#')
		archi_config['params'] = self.hash2modelparams[int(blocks[0])]
		blocks = blocks[1:]
		for i, block in enumerate(blocks):
			b_type, b_num, b_algo_id = block.split('@')
			archi_config['b{}_num'.format(i+1)] = int(b_num)
			archi_config['b{}_algo'.format(i+1)] = key_list[int(b_algo_id)]

		return archi_config

	def decode_archi_v1(self, encoded_archi):
		archi_config = {}
		key_list = self.algo_space
		blocks = encoded_archi.split('#')
		# archi_config['params'] = self.hash2modelparams[blocks[0]]
		blocks = blocks  # [1:]
		for i, block in enumerate(blocks):
			b_type, b_num, b_algo_id = block.split('@')
			archi_config['b{}_num'.format(i+1)] = int(b_num)
			archi_config['b{}_algo'.format(i+1)] = key_list[int(b_algo_id)]

		return archi_config

	def mutation_ops(self, **kwargs):
		return self.mutation_ops_v2(**kwargs)

	def mutation_ops_v1(self, **kwargs):
		# e.g. string = "42#H/3/2#V/1/13
		# # 1. mutate random state
		# def mutate_random_state(encoded_string, is_classification):
		# 	string_parts = encoded_string.split('#')
		# 	cur_rng = int(string_parts[0])
		# 	while True:
		# 		new_rng = np.random.randint(0, 100)
		# 		if cur_rng != new_rng:
		# 			break
		# 	string_parts[0] = str(new_rng)
		# 	return '#'.join(string_parts)

		# 2. mutate block
		def mutate_block(encoded_string, is_classification):
			string_parts = encoded_string.split('#')
			mutation_block_idx = np.random.randint(1, len(string_parts))
			block_parts = string_parts[mutation_block_idx].split('@')
			mutation_element_idx = np.random.randint(1, 3)
			if mutation_element_idx == 1:  # x += 1 or x -= 1
				cur_num = int(block_parts[mutation_element_idx])
				if cur_num == 5:
					block_parts[mutation_element_idx] = str(cur_num - 1)
				elif cur_num == 1:
					block_parts[mutation_element_idx] = str(cur_num + 1)
				else:
					choice = [-1, 1]
					block_parts[mutation_element_idx] = str(cur_num + choice[np.random.randint(0, 2)])
			else:  # x -> random sample from algorithm_sets
				cur_algo_idx = int(block_parts[mutation_element_idx])
				num_algos = len(self.algo_space)
				while True:
					new_algo_idx = np.random.randint(0, num_algos)
					if cur_algo_idx != new_algo_idx:
						break
				block_parts[mutation_element_idx] = str(new_algo_idx)
			string_parts[mutation_block_idx] = '@'.join(block_parts)
			return "#".join(string_parts)

		# e.g. string = "42#H/3/2#V/1/13
		self.candidate_mutation_ops = [mutate_block]
		return self.candidate_mutation_ops

	def mutate_params(self, encoded_string, is_classification, learning_tool, **kwargs):
		string_parts = encoded_string.split('#')
		new_params = learning_tool.learning_estimator.get_configuration_space().get_random_config()
		new_hash = get_hash(new_params)
		self.hash2modelparams[new_hash] = new_params
		string_parts[0] = str(new_hash)
		return '#'.join(string_parts)

	def mutation_ops_v2(self, **kwargs):
		# e.g. string = "42#H/3/2#V/1/13
		# 1. mutate params

		# 2. mutate block
		def mutate_block(encoded_string, is_classification, learning_tool, **kwargs):
			string_parts = encoded_string.split('#')
			mutation_block_idx = np.random.randint(1, len(string_parts))
			block_parts = string_parts[mutation_block_idx].split('@')
			mutation_element_idx = np.random.randint(2, 3)
			if mutation_element_idx == 1:  # x += 1 or x -= 1
				cur_num = int(block_parts[mutation_element_idx])
				if cur_num == 5:
					block_parts[mutation_element_idx] = str(cur_num - 1)
				elif cur_num == 1:
					block_parts[mutation_element_idx] = str(cur_num + 1)
				else:
					choice = [-1, 1]
					block_parts[mutation_element_idx] = str(cur_num + choice[np.random.randint(0, 2)])
			else:  # x -> random sample from algorithm_sets
				cur_algo_idx = int(block_parts[mutation_element_idx])
				num_algos = len(self.algo_space)
				while True:
					new_algo_idx = np.random.randint(0, num_algos)
					if cur_algo_idx != new_algo_idx:
						break
				block_parts[mutation_element_idx] = str(new_algo_idx)
			string_parts[mutation_block_idx] = '@'.join(block_parts)
			old_config = self.hash2modelparams[int(string_parts[0])]
			now_config = self.decode_archi('#'.join(string_parts))
			if 'params' in now_config:
				now_config.pop('params')
			new_params = learning_tool.create_learning_tool(**now_config).learning_estimator \
				.get_configuration_space().get_random_config()
			for key in old_config:  # update
				if key in new_params:
					new_params[key] = old_config[key]
			self.hash2modelparams[get_hash(new_params)] = new_params
			string_parts[0] = str(get_hash(new_params))
			return "#".join(string_parts)

		# e.g. string = "42#H/3/2#V/1/13
		self.candidate_mutation_ops = [self.mutate_params, mutate_block]
		return self.candidate_mutation_ops


if __name__ == '__main__':
	algo_space = get_all_classification_algorithm_keys()
	algo_space.remove('SVC')
	learning_tool = DeepArchiLearningTool(n_block=2, n_classes=2,
	                                      evaluation_rule='accuracy_score',
	                                      algo_space=algo_space)

