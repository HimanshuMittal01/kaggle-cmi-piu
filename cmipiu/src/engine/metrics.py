"""
Module responsible for all objective and evaluation functions
"""

import numpy as np

from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def roundoff(arr, thresholds=[0.5, 1.5, 2.5]):
    return np.where(arr < thresholds[0], 0, 
                np.where(arr < thresholds[1], 1, 
                    np.where(arr < thresholds[2], 2, 3)))

def evaluate(thresholds, y_true, y_pred):
    score = quadratic_weighted_kappa(y_true, roundoff(y_pred, thresholds))
    return -score
