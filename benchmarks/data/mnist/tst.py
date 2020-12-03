import logging
import warnings
from load_mnist_data import load_mnist
from automl.Classification.classifier import Classifier
from automl.BaseAlgorithm.classification.DeepLearningBaseAlgorithm.CNN import CNN
from keras.metrics import top_k_categorical_accuracy
from automl.performance_evaluation import top_1_accuracy
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

x_train, x_test, y_train, y_test = load_mnist()

x_train = x_train[:200]
y_train = y_train[:200]
x_test = x_test[:200]
y_test = y_test[:200]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

ans = top_1_accuracy(y_test, y_test)

print(ans)
