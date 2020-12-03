import numpy as np
import sys


def getmbof(x):
	if isinstance(x, np.ndarray):
		return "{:.2f}MB".format(x.itemsize * x.size / 1048576.0)
	return "{:.2f}MB".format(sys.getsizeof(x) / 1048576.0)
