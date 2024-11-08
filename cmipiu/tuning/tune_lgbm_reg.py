"""
Tune LGBM Regressor
"""

from functools import partial

import optuna
import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from metaflow import FlowSpec, Flow, Run, step, card

from cmipiu.metrics import evaluate, roundoff


def objective(trial, X, y, thresholds):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [500, 1000, 2000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 100, step=5),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0, step=0.1),
        "num_threads": 4,
        "verbose": -1,
    }

    skf = StratifiedKFold()
    oof_raw = np.zeros(len(y), dtype=float)  # oof predictions, before rounding
    for _, (tridx, validx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMRegressor(**param_grid)
        model.fit(X[tridx], y[tridx].to_numpy().ravel())
        oof_raw[validx] = model.predict(X[validx])

    thresholds = minimize(
        evaluate, thresholds, args=(y, oof_raw), method="Nelder-Mead"
    ).x
    y_pred_tuned = roundoff(oof_raw, thresholds=thresholds)
    return cohen_kappa_score(y, y_pred_tuned, weights="quadratic")


class OptunaLGBMFlow(FlowSpec):
    @step
    def start(self):
        self.train_run_id = Flow("ProcessTrainData").latest_successful_run.id
        self.next(self.optimization_loop)

    @step
    def optimization_loop(self):
        dataset = Run(f"ProcessTrainData/{self.train_run_id}").data.dataset["Normal"]
        self.study = optuna.create_study(
            direction="maximize", study_name="LGBM Regressor"
        )
        self.study.optimize(
            partial(
                objective,
                X=dataset["df"][dataset["features"]],
                y=dataset["df"]["sii"],
                thresholds=[0.5, 1.5, 2.5],
            ),
            n_trials=10,
        )
        self.next(self.end)

    @card
    @step
    def end(self):
        self.results = self.study.trials_dataframe()
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value


if __name__ == "__main__":
    OptunaLGBMFlow()
