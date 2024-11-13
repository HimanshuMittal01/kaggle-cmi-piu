"""Metaflow FlowSpec implementation for preparation of the training data.

MIT License (https://opensource.org/license/MIT)
Copyright (c) 2024 Himanshu Mittal
"""

import logging
from pathlib import Path

from metaflow import FlowSpec, Parameter, step, JSONType

from cmipiu.common import FeatureEngineeringSet
from cmipiu.data.ingest import (
    load_csv_data,
    clean_traincsv_data,
    get_aggregated_pq_files,
    merge_csv_pqagg_data,
    autoencode,
)
from cmipiu.data.features import feature_engineering
from cmipiu.config import CustomLogger


class ProcessTrainData(FlowSpec):
    cfg = Parameter(
        "config",
        default={
            "autoencoder": {
                "encoding_dim": 30,
                "epochs": 100,
                "learning_rate": 0.001,
            }
        },
        type=JSONType,
    )

    @property
    def logger(self) -> logging.Logger:
        """Logger for print statements within the step code."""
        logger = CustomLogger(self.__class__.__name__)
        return logger

    @step
    def start(self):
        """Branch step - separate csv and parquet data.

        Use raw version from the kaggle competition CMI-PIU 2024.
        """
        self.data_version = "child-mind-institute-problematic-internet-use/"
        self.data_dir = Path("input/raw/") / self.data_version
        self.next(self.preprocess_csv, self.preprocess_pq)

    @step
    def preprocess_csv(self):
        """Load and clean csv data.

        It drop rows that have little to no information.
        Note that this may not necessarily remove row where `sii` column is null.
        """
        # Load data
        self.logger.info("Load training data...")
        train = load_csv_data(self.data_dir / "train.csv")
        self.logger.info(f"Train csv shape (loaded): {train.shape}")

        # Clean data
        self.train = clean_traincsv_data(
            train, pq_train_dirpath=self.data_dir / "series_train.parquet"
        )
        self.logger.info(
            f"Train csv shape (after cleaning): {self.train.shape}"
        )

        self.next(self.join_csv_and_pq)

    @step
    def preprocess_pq(self):
        """Create aggregate features per user from the actigraph parquet data."""
        self.train_agg = get_aggregated_pq_files(
            self.data_dir / "series_train.parquet"
        )
        self.logger.info(f"Train pq aggregate shape: {self.train_agg.shape}")

        self.next(self.autoencode_pq)

    @step
    def autoencode_pq(self):
        """Autoencode the aggregated parquet features."""
        (
            self.train_agg_encoded,
            self.autoencoder,
            self.agg_mean,
            self.agg_std,
        ) = autoencode(
            self.train_agg,
            encoding_dim=self.cfg["autoencoder"]["encoding_dim"],
            epochs=self.cfg["autoencoder"]["epochs"],
            learning_rate=self.cfg["autoencoder"]["learning_rate"],
        )
        self.logger.info(
            f"Train pq aggregate shape (after autoencoding): {self.train_agg_encoded.shape}"
        )

        self.next(self.join_csv_and_pq)

    @step
    def join_csv_and_pq(self, inputs):
        """Join step - Merge csv and parquet streams."""
        self.merge_artifacts(inputs)

        # Join aggregates with csv data
        self.train = {
            FeatureEngineeringSet.Normal.name: merge_csv_pqagg_data(
                self.train, self.train_agg
            ),
            FeatureEngineeringSet.AutoencodedPQ.name: merge_csv_pqagg_data(
                self.train, self.train_agg_encoded
            ),
        }

        self.next(self.branch_feature_engineering)

    @step
    def branch_feature_engineering(self):
        """Branch step - Create features parallely for non-autoencoded and autoencoded data."""
        self.feature_engineering_splits = list(
            FeatureEngineeringSet.__members__.keys()
        )
        self.next(
            self.feature_engineering, foreach="feature_engineering_splits"
        )

    @step
    def feature_engineering(self):
        """Perform feature engineering for each split."""
        # Prepare dataset for training
        self.logger.info(
            f"[{self.input}] Train shape (after joining): {self.train[self.input].shape}"
        )
        self.df, self.features, self.artifacts = feature_engineering(
            self.train[self.input], training=True
        )
        self.logger.info(
            f"[{self.input}] Train shape (after feature engineering): {self.df.shape}"
        )

        self.next(self.join_feature_engineering)

    @step
    def join_feature_engineering(self, inputs):
        """Join step - Save dataset and artifacts of each feature set."""
        self.dataset = {}
        for fs in inputs:
            self.dataset[fs.input] = {
                "df": fs.df,
                "artifacts": fs.artifacts,
                "features": fs.features,
            }
            if fs.input == FeatureEngineeringSet.AutoencodedPQ.name:
                self.dataset[fs.input].update(
                    {
                        "autoencoder": fs.autoencoder,
                        "agg_mean": fs.agg_mean,
                        "agg_std": fs.agg_std,
                    }
                )

        self.next(self.end)

    @step
    def end(self):
        """End step - Does nothing."""
        pass


if __name__ == "__main__":
    ProcessTrainData()
