"""
Module for training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from cmipiu.src.engine.engine import AutoEncoder
from cmipiu.src.engine.metrics import roundoff, quadratic_weighted_kappa, evaluate
from cmipiu.src.config import config

def trainML(X, y_pciat, y, model):
    skf = StratifiedKFold()

    scores = []
    oof_raw = np.zeros(len(y), dtype=float) # oof predictions, before rounding
    oof = np.zeros(len(y), dtype=int) # oof predictions, rounded
    models = []
    for fold, (tridx, validx) in enumerate(skf.split(X, y)):
        model_ = model.clone()
        if config.use_pciat:
            model_.fit(X[tridx].to_numpy(), y_pciat[tridx].to_numpy().ravel())
        else:
            model_.fit(X[tridx].to_numpy(), y[tridx].to_numpy().ravel())
        models.append(model_)
        
        y_pred = model_.predict(X[validx].to_numpy())
        oof_raw[validx] = y_pred
        y_pred = roundoff(y_pred, thresholds=config.init_thresholds)
        oof[validx] = y_pred

        score = quadratic_weighted_kappa(y[validx].to_numpy().ravel(), y_pred)
        scores.append(score)

        accuracy = accuracy_score(y[validx].to_numpy().ravel(), y_pred)
        print(f"Fold: {fold}, Score: {score:.6f}, Accuracy: {accuracy:.6f}")
        print("-"*40)

    mean_cv_score = np.mean(scores)
    oof_score = quadratic_weighted_kappa(y, oof)

    thresholds = minimize(evaluate, config.init_thresholds, args=(y, oof_raw), method='Nelder-Mead').x
    y_pred_tuned = roundoff(oof_raw, thresholds=thresholds)

    tuned_oof_score = quadratic_weighted_kappa(y, y_pred_tuned)
    tuned_oof_accuracy = accuracy_score(y, y_pred_tuned)
    tuned_off_cm  = ConfusionMatrixDisplay.from_predictions(y, y_pred_tuned)

    with mlflow.start_run():
        mlflow.log_metric('Mean CV Score', mean_cv_score)
        mlflow.log_metric('OOF Score', oof_score)
        mlflow.log_dict({'threholds': list(thresholds)}, 'thresholds.json')
        mlflow.log_metric('Tuned OOF Score', tuned_oof_score)
        mlflow.log_metric('Tuned OOF Accuracy', tuned_oof_accuracy)
        mlflow.log_figure(tuned_off_cm.figure_, 'tuned_off_cm.png')

    return models, thresholds, oof_raw


def trainAutoEncoder(df, encoding_dim=50, epochs=100, learning_rate=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = df.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()  # For reconstruction error
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Move data to device
    inputs = df.to_torch(dtype=pl.Float32).to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        # Forward pass
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()

        # Print average loss for the epoch
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    print('Autoencoder training Complete')

    return autoencoder
