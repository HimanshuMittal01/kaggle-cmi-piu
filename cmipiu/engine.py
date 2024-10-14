"""
Module containing core algorithm and models
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor

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

            if paraminfo['model_class'] == 'EnsembleModel':
                estimators.append((paraminfo['name'], EnsembleModel(**paraminfo['params']).get_booster()))
            if paraminfo['model_class'] == 'XGBRegressor':
                estimators.append((paraminfo['name'], xgb.XGBRegressor(**paraminfo['params'])))
            elif paraminfo['model_class'] == 'LGBMRegressor':
                estimators.append((paraminfo['name'], lgb.LGBMRegressor(**paraminfo['params'])))
            elif paraminfo['model_class'] == 'RandomForestRegressor':
                estimators.append((paraminfo['name'], RandomForestRegressor(**paraminfo['params'])))
            elif paraminfo['model_class'] == 'CatBoostRegressor':
                estimators.append((paraminfo['name'], CatBoostRegressor(**paraminfo['params'])))

        model = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        return model
    
    def clone(self):
        cloned_model = EnsembleModel(self.params)
        return cloned_model

    def get_booster(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def to_params(self, name):
        return {
            'name': name,
            'model_class': 'EnsembleModel',
            'params': {'params': self.params}
        }
    
    def _ensemble_feature_importance(self, ensemble_model):
        feature_importances = []
        for estimator in ensemble_model.estimators_:
            try:
                fi = estimator.feature_importances_
            except AttributeError:
                fi = self._ensemble_feature_importance(estimator)
            
            feature_importances.append(fi)

        return np.mean(feature_importances, axis=0)

    @property
    def feature_importances_(self):
        return self._ensemble_feature_importance(self.model)
