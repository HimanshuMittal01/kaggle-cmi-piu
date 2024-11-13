"""
Module containing inference code
"""

import torch
import polars as pl

from sklearn.metrics import accuracy_score, confusion_matrix
from cmipiu.metrics import roundoff, quadratic_weighted_kappa


def predict1(model, X, y=None, thresholds=None):
    # y_preds = np.zeros((len(X), len(models)))
    # for i, model in enumerate(models):
    #     y_preds[:, i] = model.predict(X.to_numpy())

    # y_pred = y_preds.mean(axis=1)
    y_pred = model.predict(X.to_numpy())
    if thresholds is not None:
        y_pred = roundoff(y_pred, thresholds)
    return y_pred


def predictAutoEncoder(autoencoder, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = autoencoder.to(device)

    inputs = X.to_torch(dtype=pl.Float32).to(device)
    return autoencoder.encoder(inputs)


def predict2(X, model, thresholds, y_true=None):
    oof_preds = model.predict(X)
    oof_preds_rounded = roundoff(oof_preds, thresholds=thresholds)

    if y_true is not None:
        print("Score:", quadratic_weighted_kappa(y_true, oof_preds_rounded))
        print("Accuracy:", accuracy_score(y_true, oof_preds_rounded))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, oof_preds_rounded))

    return oof_preds_rounded
