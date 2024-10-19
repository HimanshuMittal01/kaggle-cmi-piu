"""
Module responsible for all objective and evaluation functions
"""

import numpy as np

from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

from cmipiu.src.config import config


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def roundoff(arr, thresholds=[0.5, 1.5, 2.5]):
    return np.where(arr < thresholds[0], 0, 
                np.where(arr < thresholds[1], 1, 
                    np.where(arr < thresholds[2], 2, 3)))

def evaluate(thresholds, y_true, y_pred):
    score = quadratic_weighted_kappa(y_true, roundoff(y_pred, thresholds))
    return -score

def evaluate_coeffs(x, y_true, y1, y2, y3):
    coeffs, thresholds = x[:3], x[3:]
    tcoeffs = sum(coeffs)
    coeffs = [x/tcoeffs for x in coeffs]
    y_pred = y1 * coeffs[0] + y2 * coeffs[1] + y3 * coeffs[2]
    score = quadratic_weighted_kappa(y_true, roundoff(y_pred, thresholds))
    return -score

def find_coeffs(y_true, oof_preds1, oof_preds2, oof_preds3):
    args = (
        y_true,
        oof_preds1,
        oof_preds2,
        oof_preds3,
    )

    best_val = 0
    coeffs = [0.33, 0.33, 0.33]
    thresholds = config.init_thresholds
    for it in range(100):
        inix = [np.random.random() for _ in range(3)] + config.init_thresholds
        res = minimize(
            evaluate_coeffs,
            inix,
            args=args,
            method='Nelder-Mead'
        )
        # show normalized x
        coeffs_, thresholds_ = res.x[:3], res.x[3:]
        tcoeffs = sum(coeffs_)
        coeffs_ = [x/tcoeffs for x in coeffs_]
        val = evaluate_coeffs(res.x, *args)
        
        if val < best_val:
            best_val = val
            coeffs = coeffs_
            thresholds = thresholds_
    
    print("Coeffs:", coeffs)
    print("Thresholds:", thresholds)
    print("Value:", -best_val)

    return coeffs, thresholds
