"""
Data Preprocessing Flow
"""

from pathlib import Path

from metaflow import FlowSpec, Flow, Run, step

from cmipiu.flows._common import FeatureEngineeringSet
from cmipiu.data.ingest import (
    load_csv_data,
    clean_testcsv_data,
    get_aggregated_pq_files,
    merge_csv_pqagg_data,
    autoencode,
)
from cmipiu.data.features import feature_engineering


class ProcessTestData(FlowSpec):
    @step
    def start(self):
        """
        Start separate streams for loading csv and parquet data
        """
        self.data_version = "child-mind-institute-problematic-internet-use/"
        self.data_dir = Path("input/raw/") / self.data_version
        self.processdata_runid = Flow("ProcessTrainData").latest_successful_run.id
        self.next(self.preprocess_csv, self.preprocess_pq)

    @step
    def preprocess_csv(self):
        """
        Load and clean csv file
        """
        # Load data
        print("Load testing data...")
        test = load_csv_data(self.data_dir / "test.csv")
        print(f"Test shape (loaded): {test.shape}")

        # Clean data
        self.test = clean_testcsv_data(test)
        print(f"Test shape (after cleaning): {self.test.shape}")

        self.next(self.join_csv_and_pq)

    @step
    def preprocess_pq(self):
        """
        Create aggregate features for parquet actigraph data
        """
        # Make aggregate
        self.test_agg = get_aggregated_pq_files(self.data_dir / "series_test.parquet")
        print(f"Test aggregate shape: {self.test_agg.shape}")

        self.next(self.autoencode_pq)

    @step
    def autoencode_pq(self):
        # Autoencode test
        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        self.test_agg_encoded, _, _, _ = autoencode(
            self.test_agg,
            autoencoder=run.data.dataset["AutoencodedPQ"]["autoencoder"],
            agg_mean=run.data.dataset["AutoencodedPQ"]["agg_mean"],
            agg_std=run.data.dataset["AutoencodedPQ"]["agg_std"],
        )
        print(f"Test aggregate shape (after autoencoding): {self.test_agg.shape}")

        self.next(self.join_csv_and_pq)

    @step
    def join_csv_and_pq(self, inputs):
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
        self.feature_engineering_splits = list(FeatureEngineeringSet.__members__.keys())
        self.next(self.feature_engineering, foreach="feature_engineering_splits")

    @step
    def feature_engineering(self):
        # Load artifacts
        run = Run(f"ProcessTrainData/{self.processdata_runid}")

        # Prepare dataset for training
        print(
            f"[{self.input}] Test shape (after joining): {self.test[self.input].shape}"
        )
        self.df, _ = feature_engineering(
            self.test[self.input],
            training=False,
            artifacts=run.data.dataset[self.input]["artifacts"],
        )
        print(
            f"[{self.input}] Train shape (after feature engineering): {self.df.shape}"
        )

        # Get only relevant features
        self.df = self.df.select(run.data.dataset[self.input]["features"])

        self.next(self.join_feature_engineering)

    @step
    def join_feature_engineering(self, inputs):
        # TODO: Data validation
        # assert train X shape and test X shape
        self.dataset = {}
        for fs in inputs:
            self.dataset[fs.input] = {
                "df": fs.df,
            }
        self.merge_artifacts(inputs, include=["processdata_runid"])
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ProcessTestData()
