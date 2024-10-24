"""
Main module containing commands
"""

from pathlib import Path

import typer
from metaflow import Runner

from cmipiu.flows.flows import end_to_end_run

app = typer.Typer()

@app.command()
def hello():
    with Runner('cmipiu/flows.py').run() as running:
        print(f'{running.run} completed')

@app.command()
def train_autoencoder_ensemble():
    print("Training Autoencoder Ensemble...")

@app.command()
def train_plain_ensemble():
    print("Training Plain Ensemble...")

@app.command()
def train_naive_ensemble():
    print("Training Naive Ensemble...")

@app.command()
def run_end_to_end():
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")
    y_test_pred = end_to_end_run(DATA_DIR)

if __name__ == '__main__':
    app()
