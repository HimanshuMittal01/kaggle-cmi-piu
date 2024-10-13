"""
Main module containing prefect workflows
"""

from pathlib import Path

import polars as pl

from cmipiu.data.cleaning import (
    handle_zero_weight_bmi,
    filter_irrelevant_data,
    make_extreme_outliers_null,
    fix_target
)
from cmipiu.data.transformation import aggregate_pq_files_v3
from cmipiu.data.features import make_XY, feature_engineering
from cmipiu.engine import XGB_LGBM_Ensemble
from cmipiu.train import trainML
from cmipiu.predict import predictML


if __name__ == '__main__':
    # Load data
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    print("Data successfully loaded!")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {train.shape}")

    # Clean data (train)
    train = (
        train
        .pipe(handle_zero_weight_bmi)
        .pipe(make_extreme_outliers_null)
        .pipe(filter_irrelevant_data, pq_train_dirpath=DATA_DIR / 'series_train.parquet')
        .pipe(fix_target)
    )
    print("Cleaning of training data completed!")
    print(f"Train shape: {train.shape}")

    # Clean data (test)
    test = (
        test
        .pipe(handle_zero_weight_bmi)
    )
    print("Cleaning of testing data completed!")
    print(f"Test shape: {test.shape}")

    # Make aggregate (train)
    files = [file for file in (DATA_DIR / 'series_train.parquet').iterdir()]
    train_agg = aggregate_pq_files_v3(files)
    print(f"Train aggregate features shape: {train_agg.shape}")

    # Make aggregate (test)
    files = [file for file in (DATA_DIR / 'series_test.parquet').iterdir()]
    test_agg = aggregate_pq_files_v3(files)
    print(f"Test aggregate features shape: {test_agg.shape}")

    # Join aggregates with main data
    train = train.join(train_agg, how='left', on='id')
    test = test.join(test_agg, how='left', on='id')
    print(f"New train shape after joining aggregates: {train.shape}")
    print(f"New test shape after joining aggregates: {test.shape}")

    # Prepare for training
    X, y_pciat, y = make_XY(train)
    print(f"Train X shape: {X.shape}")
    print(f"Train y shape: {y.shape}")
    print(f"Train y_pciat shape: {y_pciat.shape}")

    # Save intermediate output
    X.write_parquet('input/processed/X.parquet')
    y.write_parquet('input/processed/y.parquet')
    y_pciat.write_parquet('input/processed/y_pciat.parquet')

    # Feature engineering for training dataset
    X, imputer, encoder = feature_engineering(X, is_training=True)
    print(f"Train X shape after feature engineering: {X.shape}")

    # Make model
    best_params_lgbm = {
        'n_estimators': 500,
        'learning_rate': 0.012083492339234362,
        'num_leaves': 920,
        'max_depth': 11,
        'min_data_in_leaf': 180,
        'lambda_l1': 0,
        'lambda_l2': 100,
        'min_gain_to_split': 3.6624386212204185,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    best_params_xgb = {
        'n_estimators': 1000,
        'learning_rate': 0.006,
        'max_depth': 4,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'min_child_weight': 15,
        'verbosity': 0,
        'reg_alpha': 60,
        'reg_lambda': 80,
        'random_state': 42,
    }
    model = XGB_LGBM_Ensemble(
        xgb_params=best_params_xgb,
        lgbm_params=best_params_lgbm
    )
    print("Successfully built model!")

    # TRAINING
    models, thresholds = trainML(X, y_pciat, y, model)

    # INFERENCE
    X_test, _, _ = feature_engineering(test, is_training=False, imputer=imputer, encoder=encoder)
    y_pred_test = predictML(models, X=X_test, thresholds=thresholds)
    print("Inference completed!")
    print(f"First five predictions: {y_pred_test[:5]}")
