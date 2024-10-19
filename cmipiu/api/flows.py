"""
Prefect flows
"""

from pathlib import Path

from prefect import flow

@flow(log_prints=True)
def train_autoencoder_pipeline(data_dir: Path):
    pass

