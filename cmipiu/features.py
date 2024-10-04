"""
This module is responsible for all functions for feature engineering
"""

import polars as pl
import polars.selectors as cs
from tqdm import tqdm



def aggregate_pq_files(files, cols):
    def _agg_dfs(dfs):
        df = pl.concat(dfs)
        agg = df.group_by("id").agg(
            [pl.col(c).cast(pl.Float32).min().alias(f"{c}_min") for c in cols]
            + [pl.col(c).cast(pl.Float32).mean().alias(f"{c}_mean") for c in cols]
            + [pl.col(c).cast(pl.Float32).max().alias(f"{c}_max") for c in cols]
            + [pl.col(c).cast(pl.Float32).std().alias(f"{c}_std") for c in cols]
        ).collect()
        return agg
    
    aggs = []
    dfs = []
    curr_total = 0
    for pqfile in tqdm(files, desc='Aggregating pq files'):
        df = pl.scan_parquet(pqfile)
        df = df.with_columns(pl.lit(pqfile.name.removeprefix('id=')).alias('id'))
        dfs.append(df)
        _len = df.select(pl.len()).collect().item()
        curr_total += _len
        
        # Aggregate >3M rows together
        if curr_total > 3e6:
            agg = _agg_dfs(dfs)
            aggs.append(agg)

            # reset
            dfs = []
            curr_total = 0
    if dfs:
        agg = _agg_dfs(dfs)
        aggs.append(agg)

    aggdf = pl.concat(aggs)
    return aggdf


def relative_days(df):
    return df.with_columns(
        (pl.col('relative_date_PCIAT') + pl.col('time_of_day') / 86400e9).alias('day'),
        (pl.col('relative_date_PCIAT') * 86400 + pl.col('time_of_day') / 1e9).diff().fill_null(5).alias("time_diff"),
        (pl.col('time_of_day')/3600e9).round(2).alias('hour')
    )


def anglez_features(df):
    return df.with_columns(
        pl.col('anglez').abs().alias('anglez_abs'),
    ).with_columns(
        pl.col('anglez_abs').rolling_std(725).fill_null(0).alias('rolling_std_anglez')
    )


def aggregate_pq_files_v2(files):
    def _add_info_to_df(df):
        df = df.with_columns(pl.lit(pqfile.name.removeprefix('id=')).alias('id'))
        df = df.pipe(relative_days).pipe(anglez_features)
        return df
    
    def _agg_dfs(dfs):
        df = pl.concat(dfs)

        daily_avg_df = df.group_by(['id', 'relative_date_PCIAT']).agg(
            pl.col('enmo').mean().alias('daily_avg_enmo'),
            pl.col('light').mean().alias('daily_avg_light')
        )

        first7days_avg_df = df.filter(
            (pl.col('day') - pl.col('day').min()) < 7
        ).group_by(['id', 'relative_date_PCIAT']).agg(
            pl.col('enmo').mean().alias('daily_avg_enmo'),
            pl.col('light').mean().alias('daily_avg_light')
        )
        
        result_df = df.group_by('id').agg(
            pl.col('relative_date_PCIAT').min().alias('relative_start_date_PCIAT'),
            (pl.col('relative_date_PCIAT').max() - pl.col('relative_date_PCIAT').min()).alias('total_days'),
            pl.col('rolling_std_anglez').std().alias('rolling_std_anglez_abs_std'),
            pl.col('X').mean().alias('X_mean'),
            pl.col('X').std().alias('X_std'),
            pl.col('Y').mean().alias('Y_mean'),
            pl.col('Y').std().alias('Y_std'),
            pl.col('anglez').mean().alias('angleZ_mean'),
            pl.col('anglez').std().alias('angleZ_std'),
        )

        result_df = result_df.join(
            daily_avg_df.group_by('id').agg(
                pl.col('daily_avg_enmo').min().alias('daily_avg_enmo_min'),
                pl.col('daily_avg_enmo').mean().alias('daily_avg_enmo_mean'),
                pl.col('daily_avg_enmo').std().alias('daily_avg_enmo_std'),
                pl.col('daily_avg_enmo').max().alias('daily_avg_enmo_max'),
                pl.col('daily_avg_light').min().alias('daily_avg_light_min'),
                pl.col('daily_avg_light').mean().alias('daily_avg_light_mean'),
                pl.col('daily_avg_light').std().alias('daily_avg_light_std'),
                pl.col('daily_avg_light').max().alias('daily_avg_light_max'),
            ),
            on='id',
            how='left'
        ).join(
            first7days_avg_df.group_by('id').agg(
                pl.col('daily_avg_enmo').min().alias('first7_avg_enmo_min'),
                pl.col('daily_avg_enmo').mean().alias('first7_avg_enmo_mean'),
                pl.col('daily_avg_enmo').std().alias('first7_avg_enmo_std'),
                pl.col('daily_avg_enmo').max().alias('first7_avg_enmo_max'),
                pl.col('daily_avg_light').min().alias('first7_avg_light_min'),
                pl.col('daily_avg_light').mean().alias('first7_avg_light_mean'),
                pl.col('daily_avg_light').std().alias('first7_avg_light_std'),
                pl.col('daily_avg_light').max().alias('first7_avg_light_max')
            ),
            on='id',
            how='left'
        ).cast(
            {cs.numeric(): pl.Float32}
        )

        return result_df.collect()
    
    aggs = []
    dfs = []
    curr_total = 0
    for pqfile in tqdm(files, desc='Aggregating pq files'):
        df = pl.scan_parquet(pqfile)
        df = _add_info_to_df(df)
        dfs.append(df)
        _len = df.select(pl.len()).collect().item()
        curr_total += _len
        
        # Aggregate >3M rows together
        if curr_total > 3e6:
            agg = _agg_dfs(dfs)
            aggs.append(agg)

            # reset
            dfs = []
            curr_total = 0
    if dfs:
        agg = _agg_dfs(dfs)
        aggs.append(agg)

    aggdf = pl.concat(aggs)
    return aggdf
