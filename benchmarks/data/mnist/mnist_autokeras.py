import sys
sys.path.append('../../')
import time
import logging
import warnings
from load_mnist_data import load_mnist_file
import autokeras as ak
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

x_train, x_test, y_train, y_test = load_mnist_file()

x_train = x_train[:200]
y_train = y_train[:200]
x_test = x_test[:200]
y_test = y_test[:200]

print(x_train.shape, x_test.shape)
print(y_train.shape)

print("START TRAINING - - - - - - - - - - -")

if __name__ == '__main__':
	start_time = time.time()
	clf = ak.ImageClassifier()
	clf.fit(x_train, y_train, time_limit=60)
	clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
	# results = clf.predict(x_test)

	ans = clf.evaluate(x_test, y_test)
	print(ans)
	print("Time Cost: {}".format(time.time() - start_time))
