"""
Module for flows
"""

from pathlib import Path

from cmipiu.data.ingest import (
    load_csv_data,
    clean_traincsv_data,
    clean_testcsv_data,
    get_aggregated_pq_files,
)

from cmipiu.engine.metrics import find_coeffs
from cmipiu.engine.predict import predictLevel1
from cmipiu.engine.pipeline import (
    train_autoencoder_pipeline,
    train_ensemble_pipeline,
    train_naive_pipeline,
    predict_autoencoder_pipeline,
    predict_ensemble_pipeline,
    predict_naive_pipeline
)

# @flow
def end_to_end_run(data_dir: Path):
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

    # Train individual models
    output1 = train_autoencoder_pipeline(train, train_agg)
    output2 = train_ensemble_pipeline(train, train_agg)
    output3 = train_naive_pipeline(train, train_agg)

    # Find optimal coeffs and thresholds
    assert output1['y'].shape == output3['y'].shape == output3['y'].shape
    y_true = output1['y']
    coeffs, thresholds = find_coeffs(y_true, output1['oof_preds'], output2['oof_preds'], output3['oof_preds'])

    # Data checks
    print("Predicting OOF Level 1")
    predictLevel1(
        oof_preds1=output1['oof_preds'],
        oof_preds2=output2['oof_preds'],
        oof_preds3=output3['oof_preds'],
        coeffs=coeffs,
        thresholds=thresholds,
        y_true=y_true,
    )

    print("Predicting Train Level 1")
    predictLevel1(
        oof_preds1=output1['train_preds'],
        oof_preds2=output2['train_preds'],
        oof_preds3=output3['train_preds'],
        coeffs=coeffs,
        thresholds=thresholds,
        y_true=y_true,
    )

    # INFERENCE
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

    # Make predictions
    y_pred1 = predict_autoencoder_pipeline(test, test_agg, output1['autoencoder'], output1['agg_mean'], output1['agg_std'], output1['meanstd_values'], output1['imputer'], output1['encoder'], output1['models'])
    y_pred2 = predict_ensemble_pipeline(test, test_agg, output2['meanstd_values'], output2['imputer'], output2['encoder'], output2['models'])
    y_pred3 = predict_naive_pipeline(test, test_agg, output3['meanstd_values'], output3['imputer'], output3['encoder'], output3['models'])

    print("Predicting Test Level 1")
    preds = predictLevel1(
        oof_preds1=y_pred1,
        oof_preds2=y_pred2,
        oof_preds3=y_pred3,
        coeffs=coeffs,
        thresholds=thresholds,
    )

    return preds
