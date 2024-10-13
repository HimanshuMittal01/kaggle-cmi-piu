"""
Module containing inference code
"""

import numpy as np

from cmipiu.metrics import roundoff

def predictML(models, X, y=None, thresholds=[30, 50, 80]):
    y_preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        y_preds[:, i] = model.predict(X)
    
    y_pred = roundoff(y_preds.mean(axis=1), thresholds)
    return y_pred
