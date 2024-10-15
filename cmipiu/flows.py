"""
Module responsible for workflows
"""

from pathlib import Path

from prefect import flow

from cmipiu.data.ingest import (
    load_csv_data,
    clean_traincsv_data,
    clean_testcsv_data,
    get_aggregated_pq_files,
    autoencode,
    merge_csv_pqagg_data
)
from cmipiu.data.features import (
    preXY_FE,
    makeXY,
    postXY_FE,
    select_features
)
from cmipiu.engine import EnsembleModel
from cmipiu.train import trainML
from cmipiu.predict import predictML

@flow(log_prints=True)
def train_autoencoder_pipeline(data_dir: Path):
    # Load data
    print("Load training data...")
    
    train = load_csv_data(data_dir / "train.csv")
    print(f"Train shape (loaded): {train.shape}")

    # Clean data
    train = clean_traincsv_data(train, pq_train_dirpath=data_dir / 'series_train.parquet')
    print(f"Train shape (after cleaning): {train.shape}")

    # Make aggregate
    train_agg = get_aggregated_pq_files(data_dir / 'series_train.parquet')
    print(f"Train aggregate shape: {train_agg.shape}")

    # Autoencode train and test
    train_agg, autoencoder, agg_mean, agg_std = autoencode(train_agg)
    print(f"Train aggregate shape (after autoencoding): {train_agg.shape}")

    # Join aggregates with main data
    train = merge_csv_pqagg_data(train, train_agg)
    print(f"Train shape (after joining): {train.shape}")

    # Pre feature engineering for training dataset
    train, meanstd_values = preXY_FE(train, is_training=False)
    print(f"Train shape (after preXY FE): {train.shape}")

    # Prepare for training
    X, y_pciat, y = makeXY(train)
    print(f"Train X shape: {X.shape}")
    print(f"Train y shape: {y.shape}")
    print(f"Train y_pciat shape: {y_pciat.shape}")

    # Feature engineering for training dataset
    X, imputer, encoder = postXY_FE(X, is_training=True)
    print(f"Train X shape (after postXY FE): {X.shape}")

    # Make model
    params = [
        {
            'name': 'lgbm',
            'model_class': 'LGBMRegressor',
            'params': {
                'n_estimators': 300,
                'learning_rate': 0.046,
                'max_depth': 12,
                'num_leaves': 478,
                'min_data_in_leaf': 13,
                'lambda_l1': 10,
                'lambda_l2': 0.01,
                'bagging_fraction': 0.784,
                'bagging_freq': 4,
                'feature_fraction': 0.893,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'xgb',
            'model_class': 'XGBRegressor',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 5,
                'tree_method': 'exact',
                'random_state': 42,
                'verbosity': 0,
            }
        },
        {
            'name': 'catb',
            'model_class': 'CatBoostRegressor',
            'params': {
                'learning_rate': 0.05,
                'depth': 6,
                'iterations': 200,
                'l2_leaf_reg': 10,
                'verbose': 0,
                'allow_writing_files': False,
                'random_seed': 42,
            }
        },
    ]
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds = trainML(X, y_pciat, y, model)

    return {
        'autoencoder': autoencoder,
        'agg_mean': agg_mean,
        'agg_std': agg_std,
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds
    }


@flow(log_prints=True)
def predict_autoencoder_pipeline(data_dir: Path, autoencoder, agg_mean, agg_std, meanstd_values, imputer, encoder, models, thresholds):
    # Load data
    print("Load testing data...")
    
    test = load_csv_data(data_dir / "test.csv")
    print(f"Test shape (loaded): {test.shape}")

    # Clean data
    test = clean_testcsv_data(test)
    print(f"Test shape (after cleaning): {test.shape}")

    # Make aggregate
    test_agg = get_aggregated_pq_files(data_dir / 'series_test.parquet')
    print(f"Test aggregate shape: {test_agg.shape}")

    # Autoencode test
    test_agg, _, _, _ = autoencode(test_agg, autoencoder=autoencoder, agg_mean=agg_mean, agg_std=agg_std)
    print(f"Test aggregate shape (after autoencoding): {test_agg.shape}")

    # Join aggregates with main data
    test = merge_csv_pqagg_data(test, test_agg)
    print(f"Test shape (after joining): {test.shape}")

    # INFERENCE
    test, _ = preXY_FE(test, is_training=False, meanstd_values=meanstd_values)
    X_test, _, _ = postXY_FE(test, is_training=False, imputer=imputer, encoder=encoder)
    y_pred_test = predictML(models, X=X_test, thresholds=thresholds)
    print("Inference completed!")

    return y_pred_test


@flow(log_prints=True)
def run_autoencoder_pipeline(data_dir: Path):
    artifacts = train_autoencoder_pipeline(data_dir)

    output = predict_autoencoder_pipeline(
        data_dir=data_dir,
        autoencoder=artifacts['autoencoder'],
        agg_mean=artifacts['agg_mean'],
        agg_std=artifacts['agg_std'],
        meanstd_values=artifacts['meanstd_values'],
        imputer=artifacts['imputer'],
        encoder=artifacts['encoder'],
        models=artifacts['models'],
        thresholds=artifacts['thresholds']
    )

    return output


@flow(log_prints=True)
def train_ensemble_pipeline(data_dir: Path):
    # Load data
    print("Load training data...")
    
    train = load_csv_data(data_dir / "train.csv")
    print(f"Train shape (loaded): {train.shape}")

    # Clean data
    train = clean_traincsv_data(train, pq_train_dirpath=data_dir / 'series_train.parquet')
    print(f"Train shape (after cleaning): {train.shape}")

    # Make aggregate
    train_agg = get_aggregated_pq_files(data_dir / 'series_train.parquet')
    print(f"Train aggregate shape: {train_agg.shape}")

    # Join aggregates with main data
    train = merge_csv_pqagg_data(train, train_agg)
    print(f"Train shape (after joining): {train.shape}")

    # Pre feature engineering for training dataset
    train, meanstd_values = preXY_FE(train, is_training=False)
    print(f"Train shape (after preXY FE): {train.shape}")

    # Prepare for training
    X, y_pciat, y = makeXY(train)
    print(f"Train X shape: {X.shape}")
    print(f"Train y shape: {y.shape}")
    print(f"Train y_pciat shape: {y_pciat.shape}")

    # Feature engineering for training dataset
    X, imputer, encoder = postXY_FE(X, is_training=True)
    print(f"Train X shape (after postXY FE): {X.shape}")

    # Select features
    X = select_features(X)
    print(f"Train X shape (after feature selection): {X.shape}")

    # Make model
    params = [
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
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds = trainML(X, y_pciat, y, model)

    return {
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds
    }


@flow(log_prints=True)
def train_naive_pipeline(data_dir: Path):
    # Load data
    print("Load training data...")
    
    train = load_csv_data(data_dir / "train.csv")
    print(f"Train shape (loaded): {train.shape}")

    # Clean data
    train = clean_traincsv_data(train, pq_train_dirpath=data_dir / 'series_train.parquet')
    print(f"Train shape (after cleaning): {train.shape}")

    # Make aggregate
    train_agg = get_aggregated_pq_files(data_dir / 'series_train.parquet')
    print(f"Train aggregate shape: {train_agg.shape}")

    # Join aggregates with main data
    train = merge_csv_pqagg_data(train, train_agg)
    print(f"Train shape (after joining): {train.shape}")

    # Pre feature engineering for training dataset
    train, meanstd_values = preXY_FE(train, is_training=False)
    print(f"Train shape (after preXY FE): {train.shape}")

    # Prepare for training
    X, y_pciat, y = makeXY(train)
    print(f"Train X shape: {X.shape}")
    print(f"Train y shape: {y.shape}")
    print(f"Train y_pciat shape: {y_pciat.shape}")

    # Feature engineering for training dataset
    X, imputer, encoder = postXY_FE(X, is_training=True)
    print(f"Train X shape (after postXY FE): {X.shape}")

    # Make model
    params = [
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
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds = trainML(X, y_pciat, y, model)

    return {
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds
    }


