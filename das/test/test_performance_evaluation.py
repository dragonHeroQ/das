import unittest
from das.performance_evaluation import *


class TestPerformanceEvaluation(unittest.TestCase):
	def setUp(self):
		pass

	def test_initial_worst_loss(self):
		assert (initial_worst_loss('accuracy_score') == -0.0)
		assert (initial_worst_loss('mean_squared_error') == 1e82)

	def test_initial_worst_score(self):
		assert (initial_worst_score('accuracy_score') == 0.0)
		assert (initial_worst_score('mean_squared_error') == 1e82)


if __name__ == '__main__':
	unittest.main()
