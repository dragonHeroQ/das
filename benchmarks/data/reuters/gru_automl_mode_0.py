import sys
sys.path.append('../../')
import logging
import warnings
import numpy as np
from load_reuters import load_reuters
from automl.Classification.classifier import Classifier
from automl.BaseAlgorithm.classification.DeepLearningBaseAlgorithm.GRU import GRU
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


vocab_size=10000
pad_length=128

x_train, x_test, y_train, y_test = load_reuters(vocab_size,pad_length)

#you should point out the backend is tensorflow here.
#the budget type is epoch, we recommend the min and max budget to be (1, 27).
clf = Classifier(total_timebudget=3600, per_run_timebudget=3600,
                 budget_type="epoch", min_budget=1, max_budget=27,
                 automl_mode=0, classification_mode=1,
                 backend="tensorflow", evaluation_rule="accuracy_score",
                 validation_strategy="holdout", validation_strategy_args=0.8,
                 name="GRU_reuters", output_folder="./HypTuner_log")

net = GRU(num_classes=46)
clf.addClassifier({"GRU": net})

clf.fit(x_train, y_train)

config=clf.best_configs[0]
model = net.new_estimator(config)

model.fit(x_train, y_train,
          batch_size=config['batch_size'],
          epochs=9,
          verbose=1)

ans = model.evaluate(x_test, y_test)
print("[loss value,     accuracy")
print(ans)

