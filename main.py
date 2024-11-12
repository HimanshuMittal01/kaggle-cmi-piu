#!/usr/bin/env python3
"""Entrypoint script to run flows and anything easily through commands.

Usage example:

```bash
# Prepare training data
python main.py prepare-train-data

# Use --help to see all options
python main.py --help
```
"""

import typer
from metaflow import Runner
from rich import print

app = typer.Typer()


@app.command()
def prepare_train_data():
    """Run flow to prepare training data"""
    with Runner("cmipiu/flows/prepare_train_data.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def prepare_test_data():
    """Run flow to prepare testing data"""
    with Runner("cmipiu/flows/prepare_test_data.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def train():
    """Run flow to train all the models"""
    with Runner("cmipiu/flows/train_ml_models.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def predict():
    """Run inference flow"""
    with Runner("cmipiu/flows/predict_sii.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def tune(model: str):
    """Perform hypertuning on the given model"""
    if model.upper() == "LGBM_REG":
        with Runner("cmipiu/tuning/tune_lgbm_reg.py").run() as running:
            print(f"{running.run} completed")
    elif model.upper() == "XGB_REG":
        with Runner("cmipiu/tuning/tune_xgb_reg.py").run() as running:
            print(f"{running.run} completed")
    elif model.upper() == "CATBOOST_REG":
        with Runner("cmipiu/tuning/tune_catboost_reg.py").run() as running:
            print(f"{running.run} completed")


if __name__ == "__main__":
    app()
