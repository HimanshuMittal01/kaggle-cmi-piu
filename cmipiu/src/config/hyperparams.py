"""
Hyperparameters
"""


def add_hyperparameters(config):
    config.naive_params = [
        {
            'name': 'xgb1',
            'model_class': 'XGBRegressor',
            'params': {
                'verbosity': 0,
                'random_state': 42
            }
        },
        {
            'name': 'lgbm1',
            'model_class': 'LGBMRegressor',
            'params': {
                'random_state': 42,
                'verbose': -1
            }
        },
        # {
        #     'name': 'rf1',
        #     'model_class': 'RandomForestRegressor',
        #     'params': {
        #         'random_state': 42,
        #     }
        # },
        {
            'name': 'catb1',
            'model_class': 'CatBoostRegressor',
            'params': {
                'silent': True,
                'allow_writing_files': False,
                'random_state': 42,
            }
        },
    ]

    if config.use_pciat:
        config.init_thresholds = [30, 50, 80]
        
        config.autoencoder_params = [
            {
                'name': 'lgbm',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.01,
                    'max_depth': 7,
                    'num_leaves': 660,
                    'min_data_in_leaf': 30,
                    'lambda_l1': 45,
                    'lambda_l2': 25,
                    'bagging_fraction': 0.9,
                    'bagging_freq': 1,
                    'feature_fraction': 0.8,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'xgb',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.007,
                    'max_depth': 3,
                    'subsample': 0.92,
                    'colsample_bytree': 0.91,
                    'reg_alpha': 98,
                    'reg_lambda': 88,
                    'min_child_weight': 9,
                    # 'tree_method': 'exact',
                    'random_state': 42,
                    'verbosity': 0,
                }
            },
            {
                'name': 'catb',
                'model_class': 'CatBoostRegressor',
                'params': {
                    'learning_rate': 0.015,
                    'depth': 3,
                    'subsample': 0.69,
                    'colsample_bylevel': 0.565,
                    'min_data_in_leaf': 88,
                    'iterations': 400,
                    # 'l2_leaf_reg': 10,
                    'verbose': 0,
                    'allow_writing_files': False,
                    'random_seed': 42,
                }
            },
        ]
        
        config.ensemble_params = [
            {
                'name': 'lgbm1',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.0121,
                    'num_leaves': 920,
                    'max_depth': 11,
                    'min_data_in_leaf': 180,
                    'lambda_l1': 0,
                    'lambda_l2': 100,
                    'min_gain_to_split': 3.66,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 1,
                    'feature_fraction': 0.8,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'lgbm2',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.0106,
                    'num_leaves': 800,
                    'max_depth': 11,
                    'min_data_in_leaf': 55,
                    'lambda_l1': 95,
                    'lambda_l2': 55,
                    'bagging_fraction': 1,
                    'bagging_freq': 1,
                    'feature_fraction': 1.0,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'xgb1',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.019888518860232546,
                    'max_depth': 4,
                    'subsample': 0.1473684690874815,
                    'colsample_bytree': 0.738734350960037,
                    'min_child_weight': 12,
                    'verbosity': 0,
                    'reg_alpha': 39,
                    'reg_lambda': 91,
                    'random_state': 42,
                }
            },
            {
                'name': 'xgb2',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.0060719378449389984,
                    'max_depth': 4,
                    'subsample': 0.6625690509886571,
                    'colsample_bytree': 0.4296384997993591,
                    'min_child_weight': 10,
                    'verbosity': 0,
                    'reg_alpha': 89,
                    'reg_lambda': 85,
                    'random_state': 42,
                }
            },
        ]

    else:
        config.init_thresholds = [0.5, 1.5, 2.5]
        
        config.autoencoder_params = [
            {
                'name': 'lgbm',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 2000,
                    'learning_rate': 0.02,
                    'max_depth': 12,
                    'num_leaves': 1680,
                    'min_data_in_leaf': 46,
                    'lambda_l1': 20,
                    'lambda_l2': 75,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 1,
                    'feature_fraction': 1.0,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'xgb',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.018,
                    'max_depth': 6,
                    'subsample': 0.89,
                    'colsample_bytree': 0.75,
                    'reg_alpha': 34,
                    'reg_lambda': 9,
                    'random_state': 42,
                    'verbosity': 0,
                }
            },
            {
                'name': 'catb',
                'model_class': 'CatBoostRegressor',
                'params': {
                    'learning_rate': 0.015,
                    'depth': 3,
                    'subsample': 0.69,
                    'colsample_bylevel': 0.565,
                    'min_data_in_leaf': 88,
                    'iterations': 400,
                    # 'l2_leaf_reg': 10,
                    'verbose': 0,
                    'allow_writing_files': False,
                    'random_seed': 42,
                }
            },
        ]
        
        config.ensemble_params = [
            {
                'name': 'lgbm1',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.024,
                    'num_leaves': 1440,
                    'max_depth': 15,
                    'min_data_in_leaf': 50,
                    'lambda_l1': 25,
                    'lambda_l2': 20,
                    'bagging_fraction': 0.9,
                    'bagging_freq': 1,
                    'feature_fraction': 0.7,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'lgbm2',
                'model_class': 'LGBMRegressor',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.018,
                    'num_leaves': 360,
                    'max_depth': 17,
                    'min_data_in_leaf': 45,
                    'lambda_l1': 30,
                    'lambda_l2': 80,
                    'bagging_fraction': 0.9,
                    'bagging_freq': 1,
                    'feature_fraction': 0.9,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'xgb1',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.067,
                    'max_depth': 9,
                    'subsample': 0.808,
                    'colsample_bytree': 0.7,
                    'verbosity': 0,
                    'reg_alpha': 50,
                    'reg_lambda': 59,
                    'random_state': 42,
                }
            },
            {
                'name': 'xgb2',
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': 1000,
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.0435,
                    'max_depth': 8,
                    'subsample': 0.849,
                    'colsample_bytree': 0.759,
                    'verbosity': 0,
                    'reg_alpha': 44,
                    'reg_lambda': 5,
                    'random_state': 42,
                }
            },
        ]
