import sys
sys.path.append('../../')
import logging
import warnings
import tensorflow as tf
from load_cifar10 import load_cifar10
from automl.Classification.classifier import Classifier
from automl.BaseAlgorithm.classification.DeepLearningBaseAlgorithm.ResNet import ResNet
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_cifar10()

    #you should point out the backend is tensorflow here.
    #the budget type is epoch, we recommend the min and max budget to be (1, 27) or (2, 54).
    clf = Classifier(total_timebudget=3600, per_run_timebudget=3600,
                 budget_type="epoch", min_budget=1, max_budget=27,
                 automl_mode=0, classification_mode=1,
                 backend="tensorflow", evaluation_rule="top_1_accuracy",
                 validation_strategy="holdout", validation_strategy_args=0.8,
                 name="ResNet_CIFAR10", output_folder="./HypTuner_log")
    #this architechture is suitable for cifar10 and cifar100.
    #Other datasets need different resnet to reach good performance.
    net = ResNet()
    #This is ResNet20. Deeper network can be defined as ResNet(stages=[5,5,5]).
    clf.addClassifier({"ResNet": net})

    clf.fit(x_train, y_train)
    best_config = clf.best_configs[0]

    #use data augmentation to raise the test accuracy 3~4%.
    datagen=tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    datagen.fit(x_train)

    model = ResNet().new_estimator(best_config)
    #ResNet uses Batch Normalization, so batch size matters little.
    #Here it is a fixed value 128.
    batch_size = 128
    #If you do not want to use data augmentation, just use model.fit().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                      steps_per_epoch=len(x_train)/batch_size,epochs=27)
    ans = model.evaluate(x_test, y_test)
    print("[loss value,     accuracy")
    print(ans)


