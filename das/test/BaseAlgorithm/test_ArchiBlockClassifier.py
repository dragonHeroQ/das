import unittest
from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier


class TestBaseEstimator(unittest.TestCase):
	def setUp(self):
		pass

	def test_configuration_space(self):
		from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
		from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
		hbc = HorizontalBlockClassifier(2, RandomForestClassifier, e_id='h')
		hbc.get_configuration_space().show_space_names()

	def test_get_and_set_params(self):
		from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
		from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
		hbc = HorizontalBlockClassifier(2, RandomForestClassifier, e_id='h')
		print(hbc.get_params())
		params_dict = hbc.get_params()
		hbc.set_params(**params_dict)
		assert (hbc.get_params() == params_dict)


if __name__ == '__main__':
	unittest.main()
