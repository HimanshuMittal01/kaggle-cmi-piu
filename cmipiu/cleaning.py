"""
This module is responsible for data cleaning
"""

import polars as pl


def handle_zero_weight_bmi(df):
    return df.with_columns(
        pl.col('Physical-BMI').replace(0, None),
        pl.col('Physical-Weight').replace(0, None)
    )


def filter_irrelevant_data(df, pq_train_dirpath):
    pq_filenames = []
    for filename in (pq_train_dirpath).iterdir():
        pq_filenames.append(filename.name.split('=')[1])

    season_cols = [col for col in df.columns if col.endswith('Season')]
    target_cols = [col for col in df.columns if col.startswith('PCIAT')] + ['sii']
    irrelevant_cols = season_cols + target_cols + ['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Weight', 'Physical-Height', 'Physical-HeartRate', 'Physical-BMI']

    relevant_cols = [col for col in df.columns if col not in irrelevant_cols]

    return df.filter(
        ((pl.sum_horizontal(pl.col(relevant_cols).is_null()))<len(relevant_cols))
        | (pl.col('id').is_in(pq_filenames))
    ).filter(
        (pl.sum_horizontal(pl.col(target_cols).is_not_null())==len(target_cols))
        | ((pl.sum_horizontal(pl.col(relevant_cols).is_null()))<len(relevant_cols)-1)
        | (pl.col('id').is_in(pq_filenames))
    )


def make_extreme_outliers_null(df):
    return df.with_columns(
        pl.when(pl.col('id')!='83525bbe').then(pl.col('CGAS-CGAS_Score')).otherwise(None),
        pl.when(~pl.col('id').is_in(['cedf96c5', 'e252dcb6'])).then(pl.col([col for col in df.columns if col.startswith('BIA')])).otherwise(None),
    )


def fix_target(df):
    pciat_cols = [col for col in df.columns if col.startswith('PCIAT') and col!='PCIAT-PCIAT_Total' and col!='PCIAT-Season']

    return df.with_columns(
        pciat_sum=pl.sum_horizontal(pl.col(pciat_cols).is_null()),
        pciat_mean = pl.mean_horizontal(pl.col(pciat_cols)),
    ).with_columns(
        (pl.col('PCIAT-PCIAT_Total') + pl.col('pciat_mean') * pl.col('pciat_sum')).round().alias('PCIAT-PCIAT_Total')
    ).with_columns(
        sii=pl.when(
            pl.col('sii').is_not_null()
        ).then(
            pl.when(
                pl.col('PCIAT-PCIAT_Total') < 30.5
            ).then(0).otherwise(
                pl.when(
                    pl.col('PCIAT-PCIAT_Total') < 49.5
                ).then(1).otherwise(
                    pl.when(
                        pl.col('PCIAT-PCIAT_Total') < 79.5
                    ).then(2).otherwise(3)
                )
            )
        ).otherwise(None)
    ).drop(
        pl.col(['pciat_sum', 'pciat_mean'])
    )
