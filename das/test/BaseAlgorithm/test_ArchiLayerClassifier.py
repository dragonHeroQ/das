import unittest
from das.BaseAlgorithm.Classification.ArchiBlockClassifier import HorizontalBlockClassifier
from das.BaseAlgorithm.Classification.ArchiLayerClassifier import ArchiLayerClassifier


class TestBaseEstimator(unittest.TestCase):
	def setUp(self):
		pass

	def test_configuration_space(self):
		from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
		from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
		hbc = HorizontalBlockClassifier(2, RandomForestClassifier, e_id='h')
		hbc2 = HorizontalBlockClassifier(2, ExtraTreesClassifier, e_id='g')
		alc = ArchiLayerClassifier(2, [("RF", hbc), ("ERF", hbc2)], e_id=0)
		print(alc.get_model_name())
		alc.get_configuration_space().show_space_names()

	def test_get_and_set_params(self):
		from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
		from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
		hbc = HorizontalBlockClassifier(2, RandomForestClassifier)
		hbc2 = HorizontalBlockClassifier(2, ExtraTreesClassifier)
		alc = ArchiLayerClassifier(2, [("RFB", hbc), ("ERFB", hbc2)], e_id='alc')
		# print(alc.get_model_name())
		alc.get_configuration_space().show_space_names()
		params_dict = alc.get_params()
		print("params: ", params_dict)
		alc.set_params(**params_dict)
		assert (alc.get_params() == params_dict)


if __name__ == '__main__':
	unittest.main()
