"""
Data Preprocessing Flow
"""

from pathlib import Path

from metaflow import FlowSpec, step

from cmipiu._common import FeatureEngineeringSet
from cmipiu.data.ingest import (
    load_csv_data,
    clean_traincsv_data,
    get_aggregated_pq_files,
    merge_csv_pqagg_data,
    autoencode,
)
from cmipiu.data.features import feature_engineering, get_features


class ProcessTrainData(FlowSpec):
    @step
    def start(self):
        """
        Start separate streams for loading csv and parquet data.
        Using raw version from the kaggle competition CMI-PIU 2024.
        """
        self.data_version = "child-mind-institute-problematic-internet-use/"
        self.data_dir = Path("input/raw/") / self.data_version
        self.next(self.preprocess_csv, self.preprocess_pq)

    @step
    def preprocess_csv(self):
        """
        Load and clean csv file
        """
        # Load data
        print("Load training data...")
        train = load_csv_data(self.data_dir / "train.csv")
        print(f"Train csv shape (loaded): {train.shape}")

        # Clean data
        self.train = clean_traincsv_data(
            train, pq_train_dirpath=self.data_dir / "series_train.parquet"
        )
        print(f"Train csv shape (after cleaning): {self.train.shape}")

        self.next(self.join_csv_and_pq)

    @step
    def preprocess_pq(self):
        """
        Create aggregate features for parquet actigraph data
        """
        self.train_agg = get_aggregated_pq_files(self.data_dir / "series_train.parquet")
        print(f"Train pq aggregate shape: {self.train_agg.shape}")

        self.next(self.autoencode_pq)

    @step
    def autoencode_pq(self):
        """
        Autoencode aggregated parquet features
        """
        self.train_agg_encoded, self.autoencoder, self.agg_mean, self.agg_std = (
            autoencode(self.train_agg)
        )
        print(
            f"Train pq aggregate shape (after autoencoding): {self.train_agg_encoded.shape}"
        )

        self.next(self.join_csv_and_pq)

    @step
    def join_csv_and_pq(self, inputs):
        """
        Merge both streams
        """
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
        """
        Split into parallel feature set creation
        """
        self.feature_engineering_splits = list(FeatureEngineeringSet.__members__.keys())
        self.next(self.feature_engineering, foreach="feature_engineering_splits")

    @step
    def feature_engineering(self):
        """
        Perform feature engineering for each split.
        """
        # Prepare dataset for training
        print(
            f"[{self.input}] Train shape (after joining): {self.train[self.input].shape}"
        )
        self.df, self.artifacts = feature_engineering(
            self.train[self.input], training=True
        )
        print(
            f"[{self.input}] Train shape (after feature engineering): {self.df.shape}"
        )

        # Get feature columns
        self.features = get_features(self.df)

        self.next(self.join_feature_engineering)

    @step
    def join_feature_engineering(self, inputs):
        """
        Save feature dataset and artifacts with given set
        """
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
        """End step (split into train and valid if applicable)"""
        pass


if __name__ == "__main__":
    ProcessTrainData()
