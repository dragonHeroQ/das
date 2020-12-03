def load_mnist(binary_y=True):
	from tensorflow import keras
	from tensorflow.keras.datasets import mnist
	img_rows = 28
	img_cols = 28
	num_classes = 10

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# if K.image_data_format() == 'channels_first':
	# 	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	# 	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	# 	input_shape = (1, img_rows, img_cols)
	# else:
	# 	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	# 	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	# 	input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# zero-one normalization
	x_train /= 255
	x_test /= 255

	if binary_y:
		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	# input_shape = (img_rows, img_cols, 1)
	import numpy as np
	np.save("x_train.npy", x_train)
	np.save("x_test.npy", x_test)
	np.save("y_train.npy", y_train)
	np.save("y_test.npy", y_test)
	return x_train, x_test, y_train, y_test


def load_mnist_file():
	import numpy as np
	x_train = np.load("x_train.npy")
	x_test = np.load("x_test.npy")
	y_train = np.load("y_train.npy")
	y_test = np.load("y_test.npy")
	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	load_mnist(binary_y=False)

