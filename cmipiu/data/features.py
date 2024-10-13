"""
Module responsible for feature engineering
"""

import polars as pl

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer


def make_XY(df):
    print(f'Number of rows before dropping nulls: {df.shape[0]}')
    df = df.drop_nulls(subset=['sii'])
    print(f'Number of rows after dropping nulls: {df.shape[0]}')

    X = df.drop(['PCIAT-PCIAT_Total', 'sii'])
    y_pciat = df.select('PCIAT-PCIAT_Total')
    y = df.select('sii')

    return X, y_pciat, y


def feature_engineering(df, is_training=False, imputer=None, encoder=None):
    df = df.with_columns(
        (pl.col('CGAS-CGAS_Score') // 5).alias('CGAS-CGAS_Score'),
        PAQ_Total = pl.when(
            (pl.col('PAQ_C-PAQ_C_Total').is_null()) | (pl.col('PAQ_A-PAQ_A_Total').is_null())
        )
        .then((pl.col('PAQ_C-PAQ_C_Total').fill_null(0) + pl.col('PAQ_A-PAQ_A_Total').fill_null(0))/2)
        .otherwise(pl.lit(None)),
        Fitness_Endurance_Duration = pl.col('Fitness_Endurance-Time_Mins') * 60 + pl.col('Fitness_Endurance-Time_Sec')
    )
    
    # Remove all season and pciat cols
    pciat_cols = [col for col in df.columns if col.startswith('PCIAT')]
    df = df.drop(pciat_cols + ['id',
                  'PAQ_C-PAQ_C_Total', 'PAQ_A-PAQ_A_Total', 'Fitness_Endurance-Time_Mins', 
                  'Fitness_Endurance-Time_Sec'])
    
    # Encode season columns
    season_cols = [col for col in df.columns if col.endswith('Season')]
    df = df.with_columns(pl.col(season_cols).fill_null('Missing'))
    if is_training:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df[season_cols])
    else:
        assert encoder is not None

    res = encoder.transform(df[season_cols])
    encoded_season_df = pl.DataFrame(res, schema=list(encoder.get_feature_names_out()), orient="row")
    df = df.with_columns(encoded_season_df)
    
    # Impute values
    imputing_cols = [
        'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 
        'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 
        'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP', 
        'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND', 
        'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 
        'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 
        'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI', 
        'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 
        'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 
        'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
        'PreInt_EduHx-computerinternet_hoursday'
    ]
    if is_training:
        imputer = KNNImputer(n_neighbors=10, add_indicator=True)
        res = imputer.fit_transform(df[imputing_cols])
    else:
        assert imputer is not None
        res = imputer.transform(df[imputing_cols])
    
    df = df.drop(imputing_cols)
    imputed_df = pl.DataFrame(res, schema=list(imputer.get_feature_names_out()), orient="row")
    df = pl.concat([df, imputed_df], how="horizontal")
    missing_indicator_cols = [col for col in imputed_df.columns if col.startswith('missingindicator') and col not in ['missingindicator_Physical-Waist_Circumference', 'missingindicator_CGAS-CGAS_Score']]
    df = df.drop(missing_indicator_cols)
    
    drop_Features = ['daily_avg_enmo_max', 'X_0.5', 'daily_avg_light_min', 'daily_avg_enmo_std', 'enmo_0.5', 'anglez_mean', 'light_0.5', 'enmo_std', 'light_mean', 'rolling_std_anglez_std', 'enmo_0.75', 'light_0.25', 'Y_0.5', 'enmo_mean', 'daily_avg_light_std', 'total_days', 'missingindicator_CGAS-CGAS_Score', 'daily_avg_light_max', 'non-wear_flag_mean', 'daily_avg_light_mean', 'light_0.75', 'anglez_0.5', 'PAQ_A-Season']
    df = df.drop(drop_Features)
    
    if is_training:
        return df, imputer, encoder
    else:
        return df, None, None
