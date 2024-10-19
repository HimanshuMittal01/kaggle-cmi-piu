"""
Main run file
"""

from pathlib import Path

from cmipiu.src.pipeline.training import (
    train_autoencoder_pipeline,
    train_ensemble_pipeline,
    train_naive_pipeline,
    run_autoencoder_pipeline,
    run_ensemble_pipeline,
    run_naive_pipeline
)


import polars as pl
from cmipiu.src.data.ingest import load_csv_data, clean_traincsv_data
from cmipiu.src.data.features import makeXY
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from scipy.optimize import minimize
from cmipiu.src.engine.metrics import quadratic_weighted_kappa, roundoff, evaluate

if __name__ == '__main__':
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")

    output1 = train_autoencoder_pipeline(DATA_DIR)

    output2 = train_naive_pipeline(DATA_DIR)

    output3 = train_ensemble_pipeline(DATA_DIR)

    train = load_csv_data(DATA_DIR / "train.csv")
    train = clean_traincsv_data(train, pq_train_dirpath=DATA_DIR / 'series_train.parquet')
    _, y_pciat, y = makeXY(train)

    X = pl.DataFrame({
        'autoencoder_pred': output1['train_preds'],
        'naive_pred': output2['train_preds'],
        'ensemble_pred': output3['train_preds']
    })

    scores = []
    oof_raw = np.zeros(len(y), dtype=float) # oof predictions, before rounding
    oof = np.zeros(len(y), dtype=int) # oof predictions, rounded
    models = []
    skf = StratifiedKFold()
    for fold, (tridx, validx) in enumerate(skf.split(X, y)):
        # y_pred = model_.predict(X[validx].to_numpy())
        y_pred = X[validx].select(pl.concat_list(pl.all()).list.median()).to_numpy().ravel()
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

