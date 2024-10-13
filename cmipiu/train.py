"""
Module for training functions
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from cmipiu.metrics import roundoff, quadratic_weighted_kappa, evaluate


def trainML(X, y_pciat, y, model):
    skf = StratifiedKFold()

    scores = []
    oof_raw = np.zeros(len(y), dtype=float) # oof predictions, before rounding
    oof = np.zeros(len(y), dtype=int) # oof predictions, rounded
    models = []
    for fold, (tridx, validx) in enumerate(skf.split(X, y)):
        model_ = model.clone()
        model_.fit(X[tridx], y_pciat[tridx].to_numpy().ravel())
        models.append(model_)
        
        y_pred = model_.predict(X[validx])
        oof_raw[validx] = y_pred
        y_pred = roundoff(y_pred, thresholds=[30, 50, 80])
        oof[validx] = y_pred

        score = quadratic_weighted_kappa(y[validx].to_numpy().ravel(), y_pred)
        scores.append(score)

        accuracy = accuracy_score(y[validx].to_numpy().ravel(), y_pred)
        print(f"Fold: {fold}, Score: {score:.6f}, Accuracy: {accuracy:.6f}")
        print("-"*40)

    print(f"Mean score: {np.mean(scores)}")
    score = quadratic_weighted_kappa(y, oof)
    print(f"OOF score: {score}")

    thresholds = minimize(evaluate, [30, 50, 80], args=(y, oof_raw), method='Nelder-Mead').x
    print('Thresholds', thresholds)

    y_pred_tuned = roundoff(oof_raw, thresholds=thresholds)
    print("Tuned OOF Score:", quadratic_weighted_kappa(y, y_pred_tuned))

    return models, thresholds
