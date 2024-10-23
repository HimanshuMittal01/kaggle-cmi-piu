"""
Main module containing prefect workflows
"""

from pathlib import Path

import mlflow

from cmipiu import end_to_end_run

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("end-to-end-pipeline")

if __name__ == '__main__':
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")
    y_test_pred = end_to_end_run(DATA_DIR)
