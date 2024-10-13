"""
Module containing core algorithm and models
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor

class XGB_LGBM_Ensemble:
    def __init__(self, xgb_params, lgbm_params):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params

        self.model = self._build_model(self.xgb_params, self.lgbm_params)

    def _build_model(self, xgb_params, lgbm_params):
        model = VotingRegressor(
            estimators=[
                # CatBoostRegressor(silent=True, allow_writing_files=False)
                ('xgb', xgb.XGBRegressor(**xgb_params)),
                ('lgbm', lgb.LGBMRegressor(**lgbm_params))
            ],
            weights=[10, 20]
        )
        return model
    
    def clone(self):
        cloned_model = XGB_LGBM_Ensemble(
            xgb_params=self.xgb_params,
            lgbm_params=self.lgbm_params
        )
        return cloned_model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
