"""
Main module containing commands
"""

from pathlib import Path

import typer
from metaflow import Runner

app = typer.Typer()

@app.command()
def prepare_train_data():
    with Runner('cmipiu/flows/prepare_train_data.py').run() as running:
        print(f'{running.run} completed')

@app.command()
def prepare_test_data():
    with Runner('cmipiu/flows/prepare_test_data.py').run() as running:
        print(f'{running.run} completed')

@app.command()
def train():
    with Runner('cmipiu/flows/train_ml_models.py').run() as running:
        print(f'{running.run} completed')

@app.command()
def predict():
    with Runner('cmipiu/flows/predict_sii.py').run() as running:
        print(f'{running.run} completed')

if __name__ == '__main__':
    app()
