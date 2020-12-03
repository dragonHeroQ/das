import sys
sys.path.append('../../')
import logging
import warnings
from load_mnist_data import load_mnist
from automl.Classification.classifier import Classifier
from automl.BaseAlgorithm.classification.DeepLearningBaseAlgorithm.LeNet import LeNet
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

x_train, x_test, y_train, y_test = load_mnist()

#you should point out the backend is tensorflow here.
#the budget type is epoch, we recommend the min and max budget to be (1, 27).
clf = Classifier(total_timebudget=1200, per_run_timebudget=1200,
                 budget_type="epoch", min_budget=1, max_budget=27,
                 automl_mode=0, classification_mode=1,
                 backend="tensorflow", evaluation_rule="top_1_accuracy",
                 validation_strategy="holdout", validation_strategy_args=0.8,
                 name="CNN_MNIST", output_folder="./HypTuner_log")

net = LeNet(img_rows=28, img_cols=28, channels=1)
clf.addClassifier({"LeNet": net})

clf.fit(x_train, y_train)

config = clf.best_configs[0]
model = net.new_estimator(config)

model.fit(x_train, y_train,
           batch_size=config['batch_size'],
           epochs=27,
           verbose=1)

ans = model.evaluate(x_test, y_test)
print("[loss value,     accuracy")
print(ans)
