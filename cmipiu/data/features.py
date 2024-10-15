"""
Module responsible for feature engineering
"""

import polars as pl

from prefect import task

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer


@task
def preXY_FE(df, is_training=False, meanstd_values=None):
    fgc_mags = [col for col in df.columns if col.startswith('FGC') and col!='FGC-Season' and not col.endswith('Zone')]
    bia_mags = [col for col in df.columns if col.startswith('BIA') and col!='BIA-Season' and col!='BIA-BIA_Activity_Level_num' and col!='BIA-BIA_Frame_num']
    cols_to_transform = fgc_mags + bia_mags

    def make_new_wh(df):
        return df.with_columns(
            neww=pl.col('Physical-Weight') // 10,
            newh=pl.col('Physical-Height') // 6
        )
    
    def make_meanstd_cols(df, meanstd_values=None):
        if meanstd_values is None:
            return df.with_columns(
                [pl.col(col).mean().over(['Basic_Demos-Age', 'Basic_Demos-Sex', 'neww', 'newh']).alias(f'm{col}') for col in cols_to_transform]
                + [pl.col(col).std().over(['Basic_Demos-Age', 'Basic_Demos-Sex', 'neww', 'newh']).fill_null(1).alias(f's{col}') for col in cols_to_transform],
            )

        else:
            return df.join(
                meanstd_values,
                on=['Basic_Demos-Age', 'Basic_Demos-Sex', 'neww', 'newh'],
                how='left'
            )
    
    def make_tvalues(df):
        return df.with_columns(
            [((pl.col(col) - pl.col(f'm{col}')) / pl.col(f's{col}')).alias(f't{col}') for col in cols_to_transform]
        )
    
    df = (
        df
        .pipe(make_new_wh)
        .pipe(make_meanstd_cols, meanstd_values=meanstd_values)
        .pipe(make_tvalues)
    )

    tvalues = None
    if is_training:
        tvalues = df.select(
            pl.col(['Basic_Demos-Age', 'Basic_Demos-Sex', 'neww', 'newh'] + [f'm{col}' for col in cols_to_transform] + [f's{col}' for col in cols_to_transform])
        ).unique()

    df = df.drop(
        ['neww', 'newh'] + [f'm{col}' for col in cols_to_transform] + [f's{col}' for col in cols_to_transform]
    )

    return df, tvalues


@task
def makeXY(df):
    print(f'Number of rows before dropping nulls: {df.shape[0]}')
    df = df.drop_nulls(subset=['sii'])
    print(f'Number of rows after dropping nulls: {df.shape[0]}')

    X = df.drop(['PCIAT-PCIAT_Total', 'sii'])
    y_pciat = df.select('PCIAT-PCIAT_Total')
    y = df.select('sii')

    return X, y_pciat, y


@task
def postXY_FE(df, is_training=False, imputer=None, encoder=None):
    df = df.with_columns(
        (pl.col('CGAS-CGAS_Score') // 5).alias('CGAS-CGAS_Score'),
        (pl.col('Physical-Waist_Circumference').is_null().alias('missingindicator_Waist_Circumference')),
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
        imputer = KNNImputer(n_neighbors=10)
        res = imputer.fit_transform(df[imputing_cols])
    else:
        assert imputer is not None
        res = imputer.transform(df[imputing_cols])
    
    df = df.drop(imputing_cols)
    imputed_df = pl.DataFrame(res, schema=list(imputer.get_feature_names_out()), orient="row")
    df = pl.concat([df, imputed_df], how="horizontal")
    
    if is_training:
        return df, imputer, encoder
    else:
        return df, None, None


@task
def select_features(df):
    drop_features1 = ['daily_avg_enmo_std', 'light_0.25', 'enmo_0.5', 'rolling_std_anglez_0.25', 'enmo_0.75', 'non-wear_flag_mean', 'daily_avg_enmo_max', 'total_days', 'light_mean', 'tBIA-BIA_FFM', 'light_0.75', 'daily_avg_light_mean', 'daily_avg_light_max', 'enmo_std', 'Y_0.5', 'FGC-FGC_TL_Zone', 'enmo_mean', 'anglez_0.5', 'daily_avg_light_std', 'PAQ_A-Season']
    drop_features2 = ['BIA-Season', 'tBIA-BIA_TBW', 'X_0.25', 'anglez_0.75', 'rolling_std_anglez_std', 'anglez_0.25', 'FGC-Season', 'X_0.5']
    drop_features3 = ['FGC-FGC_SRR_Zone', 'rolling_std_anglez_0.5', 'BIA-BIA_FFM', 'light_0.5', 'daily_avg_light_min', 'rolling_std_anglez_0.75']
    drop_features = drop_features1 + drop_features2 + drop_features3
    df = df.drop(drop_features)

    return df
