"""
Module containing inference code
"""

import torch
import numpy as np
import polars as pl
from prefect import task

from cmipiu.metrics import roundoff

@task
def predictML(models, X, y=None, thresholds=[30, 50, 80]):
    y_preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        y_preds[:, i] = model.predict(X.to_numpy())
    
    y_pred = roundoff(y_preds.mean(axis=1), thresholds)
    return y_pred

@task
def predictAutoEncoder(autoencoder, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = autoencoder.to(device)

    inputs = X.to_torch(dtype=pl.Float32).to(device)
    return autoencoder.encoder(inputs)
