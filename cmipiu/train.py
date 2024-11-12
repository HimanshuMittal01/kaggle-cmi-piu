"""
Module for training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from cmipiu.engine.autoencoder import AutoEncoder
from cmipiu.engine.ensemble import EnsembleModel
from cmipiu.metrics import roundoff, quadratic_weighted_kappa, evaluate

from cmipiu.config import CustomLogger

logger = CustomLogger(name=__name__)


def build_model(params):
    model = EnsembleModel(params)
    return model


def train_and_evaluate_model_level1(X, y, model, init_thresholds):
    # Perform CV
    skf = StratifiedKFold()

    scores = []
    oof_raw = np.zeros(len(y), dtype=float)  # oof predictions, before rounding
    oof = np.zeros(len(y), dtype=int)  # oof predictions, rounded
    models = []
    for fold, (tridx, validx) in enumerate(skf.split(X, y)):
        model_ = model.clone()
        model_.fit(X[tridx].to_numpy(), y[tridx].to_numpy().ravel())
        models.append(model_)

        y_pred = model_.predict(X[validx].to_numpy())
        oof_raw[validx] = y_pred
        y_pred = roundoff(y_pred, thresholds=init_thresholds)
        oof[validx] = y_pred

        score = quadratic_weighted_kappa(y[validx].to_numpy().ravel(), y_pred)
        scores.append(score)

        accuracy = accuracy_score(y[validx].to_numpy().ravel(), y_pred)
        logger.info(f"Fold: {fold}, Score: {score:.6f}, Accuracy: {accuracy:.6f}")
        logger.info("-" * 40)

    # Evaluate on metrics
    cv_mean_qwk_score = np.mean(scores)
    cv_std_qwk_score = np.std(scores)
    oof_score = quadratic_weighted_kappa(y, oof)

    thresholds = minimize(
        evaluate, init_thresholds, args=(y, oof_raw), method="Nelder-Mead"
    ).x
    y_pred_tuned = roundoff(oof_raw, thresholds=thresholds)

    tuned_oof_score = quadratic_weighted_kappa(y, y_pred_tuned)
    tuned_oof_accuracy = accuracy_score(y, y_pred_tuned)
    tuned_off_cm = ConfusionMatrixDisplay.from_predictions(y, y_pred_tuned).figure_

    evaluation_metrics = {
        "CV Mean QWK Score": cv_mean_qwk_score,
        "CV Std QWK Score": cv_std_qwk_score,
        "OOF QWK Score": oof_score,
        "Tuned OOF QWK Score": tuned_oof_score,
        "Tuned OOF Accuracy": tuned_oof_accuracy,
        "Tuned OOF Confusion Matrix": tuned_off_cm,
    }

    # Retrain model on full dataset
    # Or TODO: give an option  to make mean model of fold models
    model.fit(X.to_numpy(), y.to_numpy().ravel())

    return {
        "model": model,
        "oof_raw": oof_raw,
        "evaluation_metrics": evaluation_metrics,
        "evaluation_strategy": "StratifiedKFold (5)",
        "model_strategy": "retrain_on_full_dataset",
    }


def train_and_evaluate_model_level2(X, y, model=None, init_thresholds=[0.5, 1.5, 2.5]):
    pass


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

    print("Autoencoder training Complete")

    return autoencoder
