import unittest
from das.util.common_utils import *


class TestCommonUtils(unittest.TestCase):
	def setUp(self):
		pass

	def test_search_newest_from_dir(self):
		newest_file = search_newest_from_dir('fake_dir')
		assert (newest_file == '5323.log')


if __name__ == '__main__':
	unittest.main()
