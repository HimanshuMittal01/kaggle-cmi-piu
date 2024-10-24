"""
Module responsible for workflows
"""

from cmipiu.data.ingest import (
    autoencode,
    merge_csv_pqagg_data
)
from cmipiu.data.features import (
    preXY_FE,
    makeXY,
    postXY_FE,
    select_features
)
from cmipiu.engine.engine import EnsembleModel
from cmipiu.engine.train import trainML
from cmipiu.engine.predict import predictML
from cmipiu.config import config


def train_autoencoder_pipeline(train, train_agg):
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
    params = config.autoencoder_params
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds, oof_raw = trainML(X, y_pciat, y, model)

    # INFERENCE
    y_train_preds = predictML(models, X=X, thresholds=None)

    return {
        'autoencoder': autoencoder,
        'agg_mean': agg_mean,
        'agg_std': agg_std,
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds,
        'train_preds': y_train_preds,
        'oof_preds': oof_raw,
        'X': X,
        'y_pciat': y_pciat,
        'y': y,
    }


def predict_autoencoder_pipeline(test, test_agg, autoencoder, agg_mean, agg_std, meanstd_values, imputer, encoder, models):
    # Autoencode test
    test_agg, _, _, _ = autoencode(test_agg, autoencoder=autoencoder, agg_mean=agg_mean, agg_std=agg_std)
    print(f"Test aggregate shape (after autoencoding): {test_agg.shape}")

    # Join aggregates with main data
    test = merge_csv_pqagg_data(test, test_agg)
    print(f"Test shape (after joining): {test.shape}")

    # INFERENCE
    test, _ = preXY_FE(test, is_training=False, meanstd_values=meanstd_values)
    X_test, _, _ = postXY_FE(test, is_training=False, imputer=imputer, encoder=encoder)
    y_pred_test = predictML(models, X=X_test, thresholds=None)
    print("Inference completed!")

    return y_pred_test


def train_ensemble_pipeline(train, train_agg):
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
    # X = select_features(X)
    # print(f"Train X shape (after feature selection): {X.shape}")

    # Make model
    params = config.ensemble_params
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds, oof_raw = trainML(X, y_pciat, y, model)

    # INFERENCE
    y_train_preds = predictML(models, X=X, thresholds=None)

    return {
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds,
        'train_preds': y_train_preds,
        'oof_preds': oof_raw,
        'X': X,
        'y_pciat': y_pciat,
        'y': y,
    }


def predict_ensemble_pipeline(test, test_agg, meanstd_values, imputer, encoder, models):
    # Join aggregates with main data
    test = merge_csv_pqagg_data(test, test_agg)
    print(f"Test shape (after joining): {test.shape}")

    # INFERENCE
    test, _ = preXY_FE(test, is_training=False, meanstd_values=meanstd_values)
    X_test, _, _ = postXY_FE(test, is_training=False, imputer=imputer, encoder=encoder)
    # X_test = select_features(X_test)
    y_pred_test = predictML(models, X=X_test, thresholds=None)
    print("Inference completed!")

    return y_pred_test


def train_naive_pipeline(train, train_agg):
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
    params = config.naive_params
    model = EnsembleModel(params)
    print("Successfully built model!")

    # TRAINING
    models, thresholds, oof_raw = trainML(X, y_pciat, y, model)

    # INFERENCE
    y_train_preds = predictML(models, X=X, thresholds=None)

    return {
        'meanstd_values': meanstd_values,
        'imputer': imputer,
        'encoder': encoder,
        'models': models,
        'thresholds': thresholds,
        'train_preds': y_train_preds,
        'oof_preds': oof_raw,
        'X': X,
        'y_pciat': y_pciat,
        'y': y,
    }


def predict_naive_pipeline(test, test_agg, meanstd_values, imputer, encoder, models):
    # Join aggregates with main data
    test = merge_csv_pqagg_data(test, test_agg)
    print(f"Test shape (after joining): {test.shape}")

    # INFERENCE
    test, _ = preXY_FE(test, is_training=False, meanstd_values=meanstd_values)
    X_test, _, _ = postXY_FE(test, is_training=False, imputer=imputer, encoder=encoder)
    y_pred_test = predictML(models, X=X_test, thresholds=None)
    print("Inference completed!")

    return y_pred_test
