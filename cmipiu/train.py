"""
Module for training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from prefect import task
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from cmipiu.engine import AutoEncoder
from cmipiu.metrics import roundoff, quadratic_weighted_kappa, evaluate

@task
def trainML(X, y_pciat, y, model):
    skf = StratifiedKFold()

    scores = []
    oof_raw = np.zeros(len(y), dtype=float) # oof predictions, before rounding
    oof = np.zeros(len(y), dtype=int) # oof predictions, rounded
    models = []
    for fold, (tridx, validx) in enumerate(skf.split(X, y)):
        model_ = model.clone()
        model_.fit(X[tridx].to_numpy(), y_pciat[tridx].to_numpy().ravel())
        models.append(model_)
        
        y_pred = model_.predict(X[validx].to_numpy())
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

    print("OOF Accuracy:", accuracy_score(y, y_pred_tuned))
    print("OOF Confusion matrix:")
    print(confusion_matrix(y, y_pred_tuned))

    return models, thresholds


@task
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
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()

        # Print average loss for the epoch
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    print('Autoencoder training Complete')

    return autoencoder
