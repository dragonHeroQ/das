import unittest


class TestRandomSearch(unittest.TestCase):
	def setUp(self):
		pass

	def test_get_best_record(self):
		records = [({'b1_algo': 'RandomForestClassifier', 'b1_num': 2, 'b2_algo': 'XGBClassifier', 'b2_num': 4},
		            {'loss': -0.942643391521197, 'val_accuracy_score': 0.942643391521197, 'best_nLayer': 1})]
		sorted_records = sorted(records, key=lambda x: x[1]['val_{}'.format('accuracy_score')],
		                        reverse=True)
		self.best_config = sorted_records[0][0]
		self.best_num_layers = sorted_records[0][1]['best_nLayer']
		print(self.best_config, self.best_num_layers)


if __name__ == '__main__':
	unittest.main()
