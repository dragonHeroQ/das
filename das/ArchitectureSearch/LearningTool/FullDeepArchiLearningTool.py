import das
import sys
import logging
import numpy as np
from das.ParameterSpace import *
from das.performance_evaluation import judge_rule
from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier, VerticalBlockClassifier
from das.BaseAlgorithm.Classification.ArchiLayerClassifier import ArchiLayerClassifier
from das.BaseAlgorithm.Classification.DeepArchiClassifier import DeepArchiClassifier
from das.BaseAlgorithm.Regression.ArchiBlockRegressor import HorizontalBlockRegressor, VerticalBlockRegressor
from das.BaseAlgorithm.Regression.ArchiLayerRegressor import ArchiLayerRegressor
from das.BaseAlgorithm.Regression.DeepArchiRegressor import DeepArchiRegressor
from das.BaseAlgorithm.algorithm_space import (get_algorithm_class_by_key, get_all_classification_algorithm_keys,
                                               get_all_regression_algorithm_keys)
from das.ArchitectureSearch.LearningTool.BaseLearningTool import BaseLearningTool

logger = logging.getLogger(das.logger_name)


class FullDeepArchiLearningTool(BaseLearningTool):

	def __init__(self, n_block=2, n_classes=None, evaluation_rule=None, **kwargs):
		super(FullDeepArchiLearningTool, self).__init__(n_classes=n_classes,
		                                                evaluation_rule=evaluation_rule, **kwargs)
		self.n_block = n_block
		self.learning_estimator = None
		self.last_model_storage_size = None
		self.hyper_params = None
		self.candidate_mutation_ops = None

	@property
	def is_classification(self):
		return judge_rule(self.evaluation_rule) == 'classification' and self.n_classes_ not in [None, 1]

	def set_parameter_space(self, ps=None):
		parameter_space = ParameterSpace()
		if ps is None:
			if self.is_classification:
				algo_spaces = get_all_classification_algorithm_keys()
			else:
				algo_spaces = get_all_regression_algorithm_keys()
			n_block = UniformIntSpace(name='n_block', min_val=1, max_val=2, default=1)
			random_state = UniformIntSpace(name='random_state', min_val=0, max_val=99, default=0)
			b1_type = CategorySpace(name="b1_type", choice_space=['H', 'V'])
			b1_num = UniformIntSpace(name="b1_num", min_val=1, max_val=5, default=2)
			b1_algo = CategorySpace(name="b1_algo", choice_space=algo_spaces)
			b2_type = CategorySpace(name="b2_type", choice_space=['H', 'V'])
			b2_num = UniformIntSpace(name="b2_num", min_val=1, max_val=5, default=2)
			b2_algo = CategorySpace(name="b2_algo", choice_space=algo_spaces)
			parameter_space.merge([n_block, random_state, b1_type, b1_num, b1_algo, b2_type, b2_num, b2_algo])
		else:
			tmp_space = []
			for p in ps.keys():
				ps[p].set_name(p)
				tmp_space.append(ps[p])
			parameter_space.merge(tmp_space)

		self.parameter_space = parameter_space

	def create_learning_tool(self, max_layer=0, early_stopping_rounds=1, n_folds=3, **hyper_params):
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
		for i in range(hyper_params['n_block']):
			assert 'b{}_type'.format(i+1) in hyper_params, "hyper_params = {}".format(hyper_params)
			if hyper_params['b{}_type'.format(i+1)] == 'V':
				if self.is_classification:
					block_class = VerticalBlockClassifier
				else:
					block_class = VerticalBlockRegressor
			blocks.append(("block{}".format(i),
			               block_class(nc=hyper_params['b{}_num'.format(i+1)],
			                           model_class=get_algorithm_class_by_key(
                                          hyper_params['b{}_algo'.format(i+1)]),
			                           n_classes=self.n_classes_, e_id=i, random_state=i)))
		basic_layer = layer_class(nc=len(blocks),
		                          model=blocks,
		                          n_classes=self.n_classes_,
		                          e_id=0, random_state=0)
		deep_model = model_class(base_layer=basic_layer,
		                         max_layer=max_layer,
		                         early_stopping_rounds=early_stopping_rounds,
		                         n_folds=n_folds,
		                         evaluation_rule=self.evaluation_rule,
		                         cross_validator=self.cross_validator,
		                         n_classes=self.n_classes_,
		                         e_id=hyper_params['random_state'], random_state=hyper_params['random_state'])
		self.last_model_storage_size = (sys.getsizeof(deep_model) / 1024.0)
		if self.last_model_storage_size > 5000:
			logger.warning("model size are larger than 5000 KB (5MB), please take attention!")

		# return deep_model
		return_learning_tool = FullDeepArchiLearningTool(n_block=len(blocks),
		                                                 n_classes=self.n_classes_,
		                                                 evaluation_rule=self.evaluation_rule)
		return_learning_tool.learning_estimator = deep_model
		return_learning_tool.last_model_storage_size = self.last_model_storage_size

		return return_learning_tool

	def encode_archi(self, archi_config):
		if self.is_classification:
			key_dict = get_classification_key_dict()
		else:
			key_dict = get_regression_key_dict()
		encode_string = "{}#".format(archi_config['random_state'])
		for i in range(archi_config['n_block']):
			b_type = archi_config['b{}_type'.format(i+1)]
			b_algo = archi_config['b{}_algo'.format(i+1)]
			b_num = archi_config['b{}_num'.format(i+1)]
			encode_string += "{}/{}/{}".format(b_type, b_num, key_dict[b_algo])
			if i < archi_config['n_block']-1:
				encode_string += '#'

		return encode_string

	def decode_archi(self, encoded_archi):
		archi_config = {}
		if self.is_classification:
			key_list = get_all_classification_algorithm_keys()
		else:
			key_list = get_all_regression_algorithm_keys()
		blocks = encoded_archi.split('#')
		archi_config['random_state'] = int(blocks[0])
		blocks = blocks[1:]
		archi_config['n_block'] = len(blocks)
		for i, block in enumerate(blocks):
			b_type, b_num, b_algo_id = block.split('/')
			archi_config['b{}_type'.format(i+1)] = b_type
			archi_config['b{}_num'.format(i+1)] = int(b_num)
			archi_config['b{}_algo'.format(i+1)] = key_list[int(b_algo_id)]

		return archi_config

	def mutation_ops(self):
		# e.g. string = "42#H/3/2#V/1/13
		# 1. mutate random state
		def mutate_random_state(encoded_string, is_classification):
			string_parts = encoded_string.split('#')
			cur_rng = int(string_parts[0])
			while True:
				new_rng = np.random.randint(0, 100)
				if cur_rng != new_rng:
					break
			string_parts[0] = str(new_rng)
			return '#'.join(string_parts)

		# 2. mutate block
		def mutate_block(encoded_string, is_classification):
			string_parts = encoded_string.split('#')
			mutation_block_idx = np.random.randint(1, len(string_parts))
			block_parts = string_parts[mutation_block_idx].split('/')
			mutation_element_idx = np.random.randint(0, 3)
			if mutation_element_idx == 0:  # H <-> V
				if block_parts[mutation_element_idx] == 'H':
					block_parts[mutation_element_idx] = 'V'
				else:
					block_parts[mutation_element_idx] = 'H'
			elif mutation_element_idx == 1:  # x += 1 or x -= 1
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
				if is_classification:
					num_algos = len(get_all_classification_algorithm_keys())
				else:
					num_algos = len(get_all_regression_algorithm_keys())
				while True:
					new_algo_idx = np.random.randint(0, num_algos)
					if cur_algo_idx != new_algo_idx:
						break
				block_parts[mutation_element_idx] = str(new_algo_idx)
			string_parts[mutation_block_idx] = '/'.join(block_parts)
			return "#".join(string_parts)

		# e.g. string = "42#H/3/2#V/1/13
		# 3. add/remove block
		def mutate_num_block(encoded_string, is_classification):
			def add_block():
				b_type_choice = ['H', 'V']
				b_type = b_type_choice[np.random.randint(0, 2)]
				b_num = np.random.randint(1, 6)
				if is_classification:
					num_algos = len(get_all_classification_algorithm_keys())
				else:
					num_algos = len(get_all_regression_algorithm_keys())
				b_algo_idx = np.random.randint(0, num_algos)
				block = "{}/{}/{}".format(b_type, b_num, b_algo_idx)
				string_parts.append(block)

			def remove_block():
				to_remove_block_idx = np.random.randint(1, len(string_parts))
				string_parts.pop(to_remove_block_idx)

			string_parts = encoded_string.split('#')
			cur_num_block = len(string_parts) - 1
			if cur_num_block == 4:
				remove_block()
			elif cur_num_block == 1:
				add_block()
			else:
				choice = [-1, 1]
				if choice[np.random.randint(0, 2)] == -1:
					remove_block()
				else:
					add_block()
			return "#".join(string_parts)

		self.candidate_mutation_ops = [mutate_random_state, mutate_block, mutate_num_block]
		return self.candidate_mutation_ops

