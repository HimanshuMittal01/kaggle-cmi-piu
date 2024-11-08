"""
Main module containing commands
"""

import typer
from metaflow import Runner

app = typer.Typer()


@app.command()
def prepare_train_data():
    with Runner("cmipiu/flows/prepare_train_data.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def prepare_test_data():
    with Runner("cmipiu/flows/prepare_test_data.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def train():
    with Runner("cmipiu/flows/train_ml_models.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def predict():
    with Runner("cmipiu/flows/predict_sii.py").run() as running:
        print(f"{running.run} completed")


@app.command()
def tune(model: str):
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
