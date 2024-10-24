"""
Module that defines data ingestion tasks
"""

import polars as pl

from cmipiu.data.clean import (
    handle_zero_weight_bmi,
    handle_outliers,
    filter_irrelevant_data,
    fix_target
)
from cmipiu.data.transformation import aggregate_pq_files_v3
from cmipiu.engine.predict import predictAutoEncoder
from cmipiu.engine.train import trainAutoEncoder

def load_csv_data(path):
    df = pl.read_csv(path)
    return df


def clean_traincsv_data(df, pq_train_dirpath):
    df = (
        df
        .pipe(handle_zero_weight_bmi)
        .pipe(handle_outliers)
        .pipe(filter_irrelevant_data, pq_train_dirpath=pq_train_dirpath)
        .pipe(fix_target)
    )
    return df


def clean_testcsv_data(df):
    df = (
        df
        .pipe(handle_zero_weight_bmi)
    )
    return df


def get_aggregated_pq_files(dir):
    files = [file for file in dir.iterdir()]
    train_agg = aggregate_pq_files_v3(files)
    # train_agg = load_time_series(files)
    return train_agg


def autoencode(df, autoencoder=None, agg_mean=None, agg_std=None):
    df_scaled = df.drop('id')

    if agg_mean is None:
        agg_mean = df_scaled.mean()
    if agg_std is None:
        agg_std = df_scaled.std()

    df_scaled = df_scaled.with_columns(
        [(pl.col(c) - agg_mean[c]) / (agg_std[c] + 1e-10) for c in df_scaled.columns]
    )

    if autoencoder is None:
        autoencoder = trainAutoEncoder(
            df=df_scaled,
            encoding_dim=30,
            epochs=100,
            learning_rate=0.001
        )

    df_transformed = predictAutoEncoder(autoencoder, df_scaled)
    df_transformed = pl.DataFrame(df_transformed.detach().numpy()).with_columns(df['id'])
    return df_transformed, autoencoder, agg_mean, agg_std


def merge_csv_pqagg_data(df_csv, df_pqagg):
    return df_csv.join(df_pqagg, how='left', on='id')
