"""
Main module containing prefect workflows
"""

from pathlib import Path

from cmipiu.api.flows import end_to_end_run

if __name__ == '__main__':
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")
    y_test_pred = end_to_end_run(DATA_DIR)
