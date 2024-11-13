"""Metaflow FlowSpec implementation for preparation of the testing data.

MIT License (https://opensource.org/license/MIT)
Copyright (c) 2024 Himanshu Mittal
"""

import logging
from pathlib import Path

from metaflow import FlowSpec, Flow, Run, Parameter, JSONType, step

from cmipiu.common import FeatureEngineeringSet
from cmipiu.data.ingest import (
    load_csv_data,
    clean_testcsv_data,
    get_aggregated_pq_files,
    merge_csv_pqagg_data,
    autoencode,
)
from cmipiu.data.features import feature_engineering
from cmipiu.config import CustomLogger


class ProcessTestData(FlowSpec):
    cfg = Parameter(
        "config",
        default={},
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
        self.processdata_runid = Flow(
            "ProcessTrainData"
        ).latest_successful_run.id
        self.next(self.preprocess_csv, self.preprocess_pq)

    @step
    def preprocess_csv(self):
        """Load and clean csv data.

        The cleaning is bit different from training here.
        It does not drop any rows.
        """
        # Load data
        self.logger.info("Load testing data...")
        test = load_csv_data(self.data_dir / "test.csv")
        self.logger.info(f"Test shape (loaded): {test.shape}")

        # Clean data
        self.test = clean_testcsv_data(test)
        self.logger.info(f"Test shape (after cleaning): {self.test.shape}")

        self.next(self.join_csv_and_pq)

    @step
    def preprocess_pq(self):
        """Create aggregate features per user from the actigraph parquet data."""
        # Make aggregate
        self.test_agg = get_aggregated_pq_files(
            self.data_dir / "series_test.parquet"
        )
        self.logger.info(f"Test aggregate shape: {self.test_agg.shape}")

        self.next(self.autoencode_pq)

    @step
    def autoencode_pq(self):
        """Use trained autoencoder from the training data preparation flow."""
        # Autoencode test
        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        self.test_agg_encoded, _, _, _ = autoencode(
            self.test_agg,
            autoencoder=run.data.dataset["AutoencodedPQ"]["autoencoder"],
            agg_mean=run.data.dataset["AutoencodedPQ"]["agg_mean"],
            agg_std=run.data.dataset["AutoencodedPQ"]["agg_std"],
        )
        self.logger.info(
            f"Test aggregate shape (after autoencoding): {self.test_agg.shape}"
        )

        self.next(self.join_csv_and_pq)

    @step
    def join_csv_and_pq(self, inputs):
        """Join step - Merge csv and parquet streams."""
        self.merge_artifacts(inputs)

        # Join aggregates with csv data
        self.test = {
            FeatureEngineeringSet.Normal.name: merge_csv_pqagg_data(
                self.test, self.test_agg
            ),
            FeatureEngineeringSet.AutoencodedPQ.name: merge_csv_pqagg_data(
                self.test, self.test_agg_encoded
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
        """Perform feature engineering for each split.

        Use artifacts trained while preprocessing training data.
        """
        # Load artifacts
        run = Run(f"ProcessTrainData/{self.processdata_runid}")

        # Prepare dataset for training
        self.logger.info(
            f"[{self.input}] Test shape (after joining): {self.test[self.input].shape}"
        )
        self.df, _, _ = feature_engineering(
            self.test[self.input],
            training=False,
            artifacts=run.data.dataset[self.input]["artifacts"],
        )
        self.logger.info(
            f"[{self.input}] Train shape (after feature engineering): {self.df.shape}"
        )

        # Get only relevant features from the earlier training flow
        self.features = run.data.dataset[self.input]["features"]

        self.next(self.join_feature_engineering)

    @step
    def join_feature_engineering(self, inputs):
        """Join step - Save dataset of each feature set."""
        # TODO: Data validation
        # assert train X shape and test X shape
        self.dataset = {}
        for fs in inputs:
            self.dataset[fs.input] = {"df": fs.df, "features": fs.features}
        self.merge_artifacts(inputs, include=["processdata_runid"])
        self.next(self.end)

    @step
    def end(self):
        """End step - Does nothing."""
        pass


if __name__ == "__main__":
    ProcessTestData()
