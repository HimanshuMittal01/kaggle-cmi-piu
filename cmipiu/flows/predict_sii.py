from metaflow import FlowSpec, Flow, Run, step, Parameter

from cmipiu.flows._common import ModelLevel0
from cmipiu.predict import predictML, predictLevel1

class PredictFlow(FlowSpec):
    @step
    def start(self):
        self.trainflow_runid = Flow('TrainFlow').latest_successful_run.id
        self.train_data_id1 = Run(f'TrainFlow/{self.trainflow_runid}').data.processdata_runid

        self.testdata_runid = Flow('ProcessTestData').latest_successful_run.id
        self.train_data_id2 = Run(f'ProcessTestData/{self.testdata_runid}').data.processdata_runid

        assert self.train_data_id1 == self.train_data_id2, "Train and test data could be using different artifacts"

        self.next(self.run_batch_inference)
    
    @step
    def run_batch_inference(self):
        # Find optimal coeffs and thresholds
        test_dataset = Run(f'ProcessTestData/{self.testdata_runid}').data.dataset
        trainflow = Run(f'TrainFlow/{self.trainflow_runid}').data

        # Make predictions
        y_pred1 = predictML(model=trainflow.level1_training_results[ModelLevel0.AutoencoderEnsemble.name]['model'], X=test_dataset['AutoencodedPQ']['df'])
        y_pred2 = predictML(model=trainflow.level1_training_results[ModelLevel0.PlainEnsemble.name]['model'], X=test_dataset['Normal']['df'])
        y_pred3 = predictML(model=trainflow.level1_training_results[ModelLevel0.NaiveEnsemble.name]['model'], X=test_dataset['Normal']['df'])

        self.test_preds = predictLevel1(
            oof_preds1=y_pred1,
            oof_preds2=y_pred2,
            oof_preds3=y_pred3,
            coeffs=trainflow.level2_training_results['coeffs'],
            thresholds=trainflow.level2_training_results['thresholds'],
        )

        self.next(self.end)
    
    @step
    def end(self):
        pass

if __name__ == "__main__":
    PredictFlow()
