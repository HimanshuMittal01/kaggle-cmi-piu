"""Metaflow FlowSpec implementation for training all models.

MIT License (https://opensource.org/license/MIT)
Copyright (c) 2024 Himanshu Mittal
"""

import polars as pl
from metaflow import FlowSpec, Flow, Run, step, Parameter, JSONType, card

from cmipiu.train import build_model1, train_and_evaluate1, train_and_evaluate2


class TrainFlow(FlowSpec):
    cfg = Parameter(
        "config",
        default={
            "train_models": [
                "autoencoder_ensemble",
                "plain_ensemble",
                "naive_ensemble",
            ],
            "models": {
                "autoencoder_ensemble": {
                    "fs": "AutoencodedPQ",
                    "target_col": "sii",
                    "init_thresholds": [0.5, 1.5, 2.5],
                    "params": [
                        {
                            "name": "lgbm",
                            "model_class": "LGBMRegressor",
                            "params": {
                                "n_estimators": 2000,
                                "learning_rate": 0.02,
                                "max_depth": 12,
                                "num_leaves": 1680,
                                "min_data_in_leaf": 46,
                                "lambda_l1": 20,
                                "lambda_l2": 75,
                                "bagging_fraction": 1,
                                "bagging_freq": 1,
                                "feature_fraction": 1,
                                "random_state": 42,
                                "verbose": -1,
                            },
                        },
                        {
                            "name": "xgb",
                            "model_class": "XGBRegressor",
                            "params": {
                                "n_estimators": 1000,
                                "learning_rate": 0.018,
                                "max_depth": 6,
                                "subsample": 0.89,
                                "colsample_bytree": 0.75,
                                "reg_alpha": 34,
                                "reg_lambda": 9,
                                "random_state": 42,
                                "verbosity": 0,
                            },
                        },
                        {
                            "name": "catb",
                            "model_class": "CatBoostRegressor",
                            "params": {
                                "learning_rate": 0.015,
                                "depth": 3,
                                "subsample": 0.69,
                                "colsample_bylevel": 0.565,
                                "min_data_in_leaf": 88,
                                "iterations": 400,
                                "verbose": 0,
                                "allow_writing_files": False,
                                "random_seed": 42,
                            },
                        },
                    ],
                },
                "plain_ensemble": {
                    "fs": "Normal",
                    "target_col": "sii",
                    "init_thresholds": [0.5, 1.5, 2.5],
                    "params": [
                        {
                            "name": "lgbm1",
                            "model_class": "LGBMRegressor",
                            "params": {
                                "n_estimators": 500,
                                "learning_rate": 0.024,
                                "num_leaves": 1440,
                                "max_depth": 15,
                                "min_data_in_leaf": 50,
                                "lambda_l1": 25,
                                "lambda_l2": 20,
                                "bagging_fraction": 0.9,
                                "bagging_freq": 1,
                                "feature_fraction": 0.7,
                                "random_state": 42,
                                "verbose": -1,
                            },
                        },
                        {
                            "name": "lgbm2",
                            "model_class": "LGBMRegressor",
                            "params": {
                                "n_estimators": 1000,
                                "learning_rate": 0.018,
                                "num_leaves": 360,
                                "max_depth": 17,
                                "min_data_in_leaf": 45,
                                "lambda_l1": 30,
                                "lambda_l2": 80,
                                "bagging_fraction": 0.9,
                                "bagging_freq": 1,
                                "feature_fraction": 0.9,
                                "random_state": 42,
                                "verbose": -1,
                            },
                        },
                        {
                            "name": "xgb1",
                            "model_class": "XGBRegressor",
                            "params": {
                                "n_estimators": 1000,
                                "objective": "reg:squarederror",
                                "learning_rate": 0.067,
                                "max_depth": 9,
                                "subsample": 0.808,
                                "colsample_bytree": 0.7,
                                "verbosity": 0,
                                "reg_alpha": 50,
                                "reg_lambda": 59,
                                "random_state": 42,
                            },
                        },
                        {
                            "name": "xgb2",
                            "model_class": "XGBRegressor",
                            "params": {
                                "n_estimators": 1000,
                                "objective": "reg:squarederror",
                                "learning_rate": 0.0435,
                                "max_depth": 8,
                                "subsample": 0.849,
                                "colsample_bytree": 0.759,
                                "verbosity": 0,
                                "reg_alpha": 44,
                                "reg_lambda": 5,
                                "random_state": 42,
                            },
                        },
                    ],
                },
                "naive_ensemble": {
                    "fs": "Normal",
                    "target_col": "sii",
                    "init_thresholds": [0.5, 1.5, 2.5],
                    "params": [
                        {
                            "name": "xgb1",
                            "model_class": "XGBRegressor",
                            "params": {"verbosity": 0, "random_state": 42},
                        },
                        {
                            "name": "lgbm1",
                            "model_class": "LGBMRegressor",
                            "params": {"random_state": 42, "verbose": -1},
                        },
                        {
                            "name": "catb1",
                            "model_class": "CatBoostRegressor",
                            "params": {
                                "silent": True,
                                "allow_writing_files": False,
                                "random_state": 42,
                            },
                        },
                    ],
                },
            },
        },
        type=JSONType,
    )

    @step
    def start(self):
        """Branch step - Split into parallel training for list of `train_models`."""
        self.processdata_runid = Flow(
            "ProcessTrainData"
        ).latest_successful_run.id
        self.model_names = self.cfg["train_models"]
        self.next(self.train_model1, foreach="model_names")

    @card
    @step
    def train_model1(self):
        """Build and train weights for each model."""
        # Initialize model from config parameters
        model = build_model1(self.cfg[self.input]["params"])

        # This performs CV evaluation and then retrain on the full dataset
        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        dataset = run.data.dataset[self.cfg[self.input]["fs"]]
        self.results = train_and_evaluate1(
            X=dataset["df"][dataset["features"]],
            y=dataset["df"][self.cfg[self.input]["target_col"]],
            model=model,
            init_thresholds=self.cfg[self.input]["init_thresholds"],
        )

        self.next(self.join_all_models)

    @step
    def join_all_models(self, inputs):
        """Join step - Save training results of all models."""
        self.results1 = {}
        for input in inputs:
            self.results1[input.input] = input.results

        self.merge_artifacts(inputs, include=["processdata_runid"])
        self.next(self.train_model2)

    @step
    def train_model2(self):
        """Ensemble level 1 models using Linear Regression."""
        # Concat three model predictions into a dataset
        level1_oof_preds = {}
        for model_name in self.results1:
            level1_oof_preds[model_name] = self.results1[model_name]["oof_raw"]
        df_oof_preds1 = pl.DataFrame(level1_oof_preds)

        # Train and Evaluate linear regressoin model
        # Final y_true must be 'sii' and init thresholds is also fixed
        # TODO: Data validation of target shape (for ensembling)
        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        dataset = run.data.dataset[self.cfg[model_name]["fs"]]
        y_true = dataset["df"]["sii"]
        self.results = train_and_evaluate2(
            X=df_oof_preds1, y=y_true, init_thresholds=[0.5, 1.5, 2.5]
        )

        self.next(self.end)

    @step
    def end(self):
        """End step - Save model and optimal threshold only for inference later."""
        self.results2 = {
            "model": self.results["model"],
            "thresholds": self.results["thresholds"],
        }


if __name__ == "__main__":
    TrainFlow()
