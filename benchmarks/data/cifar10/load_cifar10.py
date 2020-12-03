import sys
sys.path.append('../../')
import tensorflow as tf

def load_cifar10():
	img_rows = 32
	img_cols = 32
	num_classes = 10

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# standard normalization
	x_train = x_train / 255 * 2 - 1
	x_test = x_test / 255 * 2 - 1

	# convert class vectors to binary class matrices
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

	return x_train, x_test, y_train, y_test
