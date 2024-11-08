"""
Module for flows
"""

import yaml

from metaflow import FlowSpec, Flow, Run, step, Parameter, IncludeFile, card

from cmipiu.flows._common import ModelLevel0
from cmipiu.train import build_model, train_and_evaluate_model_level1
from cmipiu.metrics import find_coeffs


class TrainFlow(FlowSpec):
    train_config = IncludeFile("train_config_path", default="config/train.yaml")
    model_names = Parameter(
        "model_names",
        default=",".join([model_name for model_name in ModelLevel0.__members__.keys()]),
        separator=",",
    )

    @step
    def start(self):
        """
        Load train configuration and parallel train given models in the new step
        """
        train_config = yaml.safe_load(self.train_config)
        self.train_parameters = {
            model.name: train_config[model.value] for model in ModelLevel0
        }

        for model_name in self.model_names:
            assert model_name in self.train_parameters, f"Undefined model: {model_name}"

        self.processdata_runid = Flow("ProcessTrainData").latest_successful_run.id
        self.next(self.train_model_level1, foreach="model_names")

    @card
    @step
    def train_model_level1(self):
        with open(self.train_parameters[self.input]["params_path"]) as f:
            self.model_params = yaml.safe_load(f)

        model = build_model(self.model_params["params"])

        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        dataset = run.data.dataset[self.train_parameters[self.input]["fs"]]
        self.training_results = train_and_evaluate_model_level1(
            X=dataset["df"][dataset["features"]],
            y=dataset["df"][self.train_parameters[self.input]["target_col"]],
            model=model,
            init_thresholds=self.train_parameters[self.input]["init_thresholds"],
        )

        self.next(self.join_all_models)

    @step
    def join_all_models(self, inputs):
        self.level1_training_results = {}
        for input in inputs:
            self.level1_training_results[input.input] = input.training_results

        self.merge_artifacts(inputs, include=["processdata_runid", "train_parameters"])
        self.next(self.train_model_level2)

    @step
    def train_model_level2(self):
        level1_oof_preds = []
        for model_name in self.level1_training_results:
            level1_oof_preds.append(self.level1_training_results[model_name]["oof_raw"])

        run = Run(f"ProcessTrainData/{self.processdata_runid}")
        dataset = run.data.dataset[self.train_parameters[model_name]["fs"]]

        # Final y_true must be 'sii' and init thresholds is also fixed
        # TODO: Data validation of target shape (for ensembling)
        self.coeffs, self.thresholds = find_coeffs(
            dataset["df"]["sii"], *level1_oof_preds, [0.5, 1.5, 2.5]
        )

        # Data checks
        # print("Predicting OOF Level 1")
        # predictLevel1(
        #     oof_preds1=output1['oof_preds'],
        #     oof_preds2=output2['oof_preds'],
        #     oof_preds3=output3['oof_preds'],
        #     coeffs=coeffs,
        #     thresholds=thresholds,
        #     y_true=y_true,
        # )

        # print("Predicting Train Level 1")
        # predictLevel1(
        #     oof_preds1=output1['train_preds'],
        #     oof_preds2=output2['train_preds'],
        #     oof_preds3=output3['train_preds'],
        #     coeffs=coeffs,
        #     thresholds=thresholds,
        #     y_true=y_true,
        # )

        self.next(self.end)

    @step
    def end(self):
        self.level2_training_results = {
            "coeffs": self.coeffs,
            "thresholds": self.thresholds,
        }


if __name__ == "__main__":
    TrainFlow()
