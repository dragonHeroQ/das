import numpy as np

a = np.ones((200000, 30000), dtype=np.float64)


def getGBSize(x):
	return x.itemsize * x.size / 1048576.0 / 1024.0


print(getGBSize(a))

