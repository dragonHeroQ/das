import sys
import pickle
sys.path.append("../")
sys.path.append("../../")
import numpy as np
from automl.compmodel import CompositeModel
from automl.crossvalidate import cross_validate_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
from airfoil.load_airfoil import load_airfoil
from automl.get_algorithm import get_algorithm_by_key
import importlib


mse_dict = pickle.load(open('{}_mses_{}.pkl'.format("superconduct", 0), 'rb'))

print(mse_dict['Time_Cost'])
