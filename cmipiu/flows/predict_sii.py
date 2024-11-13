from metaflow import FlowSpec, Flow, Run, step

from cmipiu.common import ModelLevel0
from cmipiu.config import CustomLogger
from cmipiu.predict import predictML, predictLevel1


class PredictFlow(FlowSpec):
    @property
    def logger(self):
        """
        Get the logger for this class
        """
        logger = CustomLogger(name=self.__class__.__name__)
        return logger

    @step
    def start(self):
        self.logger.warning("My named is Khan")

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
        # Find optimal coeffs and thresholds
        test_dataset = Run(
            f"ProcessTestData/{self.testdata_runid}"
        ).data.dataset
        trainflow = Run(f"TrainFlow/{self.trainflow_runid}").data

        # Make predictions
        y_pred1 = predictML(
            model=trainflow.results1[ModelLevel0.AutoencoderEnsemble.name][
                "model"
            ],
            X=test_dataset["AutoencodedPQ"]["df"],
        )
        y_pred2 = predictML(
            model=trainflow.results1[ModelLevel0.PlainEnsemble.name]["model"],
            X=test_dataset["Normal"]["df"],
        )
        y_pred3 = predictML(
            model=trainflow.results1[ModelLevel0.NaiveEnsemble.name]["model"],
            X=test_dataset["Normal"]["df"],
        )

        self.test_preds = predictLevel1(
            oof_preds1=y_pred1,
            oof_preds2=y_pred2,
            oof_preds3=y_pred3,
            coeffs=trainflow.results2["coeffs"],
            thresholds=trainflow.results2["thresholds"],
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PredictFlow()
