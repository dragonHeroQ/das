import unittest
from das.BaseAlgorithm.Classification.ArchiLayerClassifier import ArchiLayerClassifier
from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier


class TestBaseEstimator(unittest.TestCase):
	def setUp(self):
		pass

	def test_set_random_state(self):
		est = ArchiLayerClassifier(nc=2, model=[("EXT", ExtraTreesClassifier()), ("EXT", ExtraTreesClassifier())], c_id=0)
		print(est.get_model_name())
		print(est.get_params(deep=True, only_cfg=False))


if __name__ == '__main__':
	unittest.main()
