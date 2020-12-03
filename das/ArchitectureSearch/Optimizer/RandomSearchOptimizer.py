from das.ArchitectureSearch.Optimizer.BaseOptimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):

    def __init__(self, parameter_space):
        super(RandomSearchOptimizer, self).__init__(parameter_space=parameter_space)

    def get_next_config(self, debug=False):
        if debug:
            return self.get_debug_config()
        return self.parameter_space.get_random_config()

    def get_debug_config(self):
        return {'b1_algo': 'LGBClassifier', 'b1_num': 4, 'b2_algo': 'MLPClassifier', 'b2_num': 1}
