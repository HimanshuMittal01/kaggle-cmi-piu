"""Metaflow FlowSpec implementation for inference flow.

MIT License (https://opensource.org/license/MIT)
Copyright (c) 2024 Himanshu Mittal
"""

import logging

import polars as pl
from metaflow import FlowSpec, Flow, Run, step

from cmipiu.common import ModelLevel0
from cmipiu.config import CustomLogger
from cmipiu.predict import predict1, predict2


class PredictFlow(FlowSpec):
    @property
    def logger(self) -> logging.Logger:
        """Logger for print statements within the step code."""
        logger = CustomLogger(self.__class__.__name__)
        return logger

    @step
    def start(self):
        """Start step - Load model and testing data from `TrainFlow` and `ProcessTestData`."""
        self.trainflow_runid = Flow("TrainFlow").latest_successful_run.id
        self.train_data_id1 = Run(
            f"TrainFlow/{self.trainflow_runid}"
        ).data.processdata_runid

        self.testdata_runid = Flow("ProcessTestData").latest_successful_run.id
        self.train_data_id2 = Run(
            f"ProcessTestData/{self.testdata_runid}"
        ).data.processdata_runid

        assert (
            self.train_data_id1 == self.train_data_id2
        ), "Train and test data could be using different artifacts"

        self.next(self.run_batch_inference)

    @step
    def run_batch_inference(self):
        """Predict using final model.

        First, predict using all three level 1 models.
        """
        # Find optimal coeffs and thresholds
        test_dataset = Run(
            f"ProcessTestData/{self.testdata_runid}"
        ).data.dataset
        trainflow = Run(f"TrainFlow/{self.trainflow_runid}").data

        # Make level 1 predictions
        level1_preds = {
            ModelLevel0.AutoencoderEnsemble.value: predict1(
                model=trainflow.results1[ModelLevel0.AutoencoderEnsemble.value][
                    "model"
                ],
                X=test_dataset["AutoencodedPQ"]["df"].select(
                    test_dataset["AutoencodedPQ"]["features"]
                ),
            ),
            ModelLevel0.PlainEnsemble.value: predict1(
                model=trainflow.results1[ModelLevel0.PlainEnsemble.value][
                    "model"
                ],
                X=test_dataset["Normal"]["df"].select(
                    test_dataset["Normal"]["features"]
                ),
            ),
            ModelLevel0.NaiveEnsemble.value: predict1(
                model=trainflow.results1[ModelLevel0.NaiveEnsemble.value][
                    "model"
                ],
                X=test_dataset["Normal"]["df"].select(
                    test_dataset["Normal"]["features"]
                ),
            ),
        }
        df_oof_preds1 = pl.DataFrame(level1_preds)

        # Make level 2 predictions
        self.test_preds = predict2(
            X=df_oof_preds1,
            model=trainflow.results2["model"],
            thresholds=trainflow.results2["thresholds"],
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PredictFlow()
