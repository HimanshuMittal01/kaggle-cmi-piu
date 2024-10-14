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
from cmipiu.data.features import preXY_FE, makeXY, postXY_FE, select_features
from cmipiu.engine import EnsembleModel
from cmipiu.train import trainML
from cmipiu.predict import predictML

def build_model():
    params1 = [
        {
            'name': 'lgbm1',
            'model_class': 'LGBMRegressor',
            'params': {
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
        },
        {
            'name': 'lgbm2',
            'model_class': 'LGBMRegressor',
            'params': {
                'n_estimators': 1000,
                'learning_rate': 0.010408028856708502,
                'num_leaves': 200,
                'max_depth': 3,
                'min_data_in_leaf': 180,
                'lambda_l1': 55,
                'lambda_l2': 35,
                'min_gain_to_split': 12.248451136883414,
                'bagging_fraction': 0.9000000000000001,
                'bagging_freq': 1,
                'feature_fraction': 0.8,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'xgb1',
            'model_class': 'XGBRegressor',
            'params': {
                'n_estimators': 1000,
                'objective': 'reg:squarederror',
                'learning_rate': 0.019888518860232546,
                'max_depth': 4,
                'subsample': 0.1473684690874815,
                'colsample_bytree': 0.738734350960037,
                'min_child_weight': 12,
                'verbosity': 0,
                'reg_alpha': 39,
                'reg_lambda': 91,
                'random_state': 42,
            }
        },
        {
            'name': 'xgb2',
            'model_class': 'XGBRegressor',
            'params': {
                'n_estimators': 1000,
                'objective': 'reg:squarederror',
                'learning_rate': 0.0060719378449389984,
                'max_depth': 4,
                'subsample': 0.6625690509886571,
                'colsample_bytree': 0.4296384997993591,
                'min_child_weight': 10,
                'verbosity': 0,
                'reg_alpha': 89,
                'reg_lambda': 85,
                'random_state': 42,
            }
        },
    ]

    params2 = [
        {
            'name': 'xgb1',
            'model_class': 'XGBRegressor',
            'params': {
                'verbosity': 0,
                'random_state': 42
            }
        },
        {
            'name': 'lgbm1',
            'model_class': 'LGBMRegressor',
            'params': {
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'rf1',
            'model_class': 'RandomForestRegressor',
            'params': {
                'random_state': 42,
            }
        },
        {
            'name': 'catb1',
            'model_class': 'CatBoostRegressor',
            'params': {
                'silent': True,
                'allow_writing_files': False,
                'random_state': 42,
            }
        },
    ]

    model = EnsembleModel(
        [
            EnsembleModel(params1).to_params('ensemble1'),
            EnsembleModel(params2).to_params('ensemble2')
        ]
    )
    return model


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
    print(f"Train shape after joining aggregates: {train.shape}")
    print(f"Test shape after joining aggregates: {test.shape}")

    # Pre feature engineering for training dataset
    train, meanstd_values = preXY_FE(train, is_training=False)
    print(f"Train shape after preXY FE: {train.shape}")

    # Prepare for training
    X, y_pciat, y = makeXY(train)
    print(f"Train X shape: {X.shape}")
    print(f"Train y shape: {y.shape}")
    print(f"Train y_pciat shape: {y_pciat.shape}")

    # Save intermediate output
    X.write_parquet('input/processed/X.parquet')
    y.write_parquet('input/processed/y.parquet')
    y_pciat.write_parquet('input/processed/y_pciat.parquet')

    # Feature engineering for training dataset
    X, imputer, encoder = postXY_FE(X, is_training=True)
    print(f"Train X shape after feature engineering: {X.shape}")

    # Select features
    X = select_features(X)
    print(f"Train X shape after selecting features: {X.shape}")

    # Make model
    model = build_model()
    print("Successfully built model!")

    # TRAINING
    models, thresholds = trainML(X, y_pciat, y, model)

    # INFERENCE
    test, _ = preXY_FE(test, is_training=False, meanstd_values=meanstd_values)
    X_test, _, _ = postXY_FE(test, is_training=False, imputer=imputer, encoder=encoder)
    X_test = select_features(X_test)
    y_pred_test = predictML(models, X=X_test, thresholds=thresholds)
    print("Inference completed!")
    print(f"First five predictions: {y_pred_test[:5]}")
