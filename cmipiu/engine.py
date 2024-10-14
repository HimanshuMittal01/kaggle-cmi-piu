"""
Module containing core algorithm and models
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor

class EnsembleModel:
    def __init__(self, params):
        self.params = params
        self.model = self._build_model(self.params)

    def _build_model(self, params):
        estimators = []
        weights = []
        for paraminfo in params:
            if 'weight' in paraminfo:
                weights.append(paraminfo['weight'])
            else:
                weights.append(1)

            if paraminfo['model_class'] == 'XGBRegressor':
                estimators.append((paraminfo['name'], xgb.XGBRegressor(**paraminfo['params'])))
            elif paraminfo['model_class'] == 'LGBMRegressor':
                estimators.append((paraminfo['name'], lgb.LGBMRegressor(**paraminfo['params'])))
            # elif paraminfo['model_class'] == 'CatBoostRegressor':
            #     estimators.append((paraminfo['name'], paraminfo['params']))
            #     CatBoostRegressor(silent=True, allow_writing_files=False)

        model = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        return model
    
    def clone(self):
        cloned_model = EnsembleModel(self.params)
        return cloned_model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_booster(self):
        return self.model
