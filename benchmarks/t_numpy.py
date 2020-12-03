import numpy as np
algo_space = ["SGDClassifier",
              "GBDTClassifier",
              "LogisticRegression",
              "DecisionTreeClassifier",
              "RandomForestClassifier",
              "SVC",
              "AdaboostClassifier",
              "LinearDiscriminantAnalysis",
              "QuadraticDiscriminantAnalysis",
              "ExtraTreesClassifier",
              "BernoulliNB",
              "MultinomialNB",
              "GaussianNB",
              # "GPClassifier",
              "MLPClassifier",
              "KNeighborsClassifier",
              # "RadiusNeighborsClassifier",
              "XGBClassifier",
              "LGBClassifier",
              # "IdentityClassifier"
              ]
from das.BaseAlgorithm.algorithm_space import get_algorithm_class_by_key
len_algo_space = len(algo_space)
res = np.random.choice(range(len_algo_space), 4)
res = list(map(lambda x: get_algorithm_class_by_key(algo_space[x]), res))
print(res)
