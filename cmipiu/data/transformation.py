"""
This module is responsible for all functions for feature engineering
"""

from concurrent.futures import ThreadPoolExecutor

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


def aggregate_pq_files_v3(files):
    def _add_info_to_df(df):
        df = df.with_columns(pl.lit(pqfile.name.removeprefix('id=')).alias('id'))
        df = df.pipe(relative_days).pipe(anglez_features)
        return df
    
    def make_feature_exps(col):
        return [
            pl.col(col).min().alias(f'{col}_min'),
            pl.col(col).max().alias(f'{col}_max'),
            pl.col(col).mean().alias(f'{col}_mean'),
            pl.col(col).std().alias(f'{col}_std'),
        ] + [pl.col(col).quantile(q).alias(f'{col}_{q}') for q in [0.25,0.5,0.75]]
    
    def _agg_dfs(dfs):
        df = pl.concat(dfs)

        daily_avg_df = df.group_by(['id', 'relative_date_PCIAT']).agg(
            pl.col('enmo').mean().alias('daily_avg_enmo'),
            pl.col('light').mean().alias('daily_avg_light')
        )

        result_df = df.group_by('id').agg(
            pl.col('non-wear_flag').mean().alias('non-wear_flag_mean'),
            pl.col('non-wear_flag').std().alias('non-wear_flag_std'),
            pl.col('quarter').mean().alias('quarter_mean'),
            pl.col('weekday').mean().alias('weekday_mean'),
            pl.col('weekday').std().alias('weekday_std'),
            pl.col('day').mean().alias('day_mean'),
            pl.col('day').std().alias('day_std'),
            pl.col('time_diff').max().alias('time_diff_max'),
            *make_feature_exps('X'),
            *make_feature_exps('Y'),
            *make_feature_exps('Z'),
            *make_feature_exps('enmo'),
            *make_feature_exps('light'),
            *make_feature_exps('rolling_std_anglez'),
            *make_feature_exps('relative_date_PCIAT'),
            *make_feature_exps('time_of_day'),
        )

        result_df = result_df.join(
            daily_avg_df.group_by('id').agg(
                pl.col('daily_avg_enmo').min().alias('daily_avg_enmo_min'),
                pl.col('daily_avg_enmo').mean().alias('daily_avg_enmo_mean'),
                pl.col('daily_avg_enmo').std().alias('daily_avg_enmo_std'),
                pl.col('daily_avg_enmo').max().alias('daily_avg_enmo_max'),
                pl.col('daily_avg_light').max().alias('daily_avg_light_max'),
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

    # Filter out days having total days <= 0
    aggdf = aggdf.filter(
        (pl.col('relative_date_PCIAT_max') - pl.col('relative_date_PCIAT_min'))>0
    )
    return aggdf


def process_file(filename):
    df = pl.read_parquet(filename/'part-0.parquet')
    df = df.drop('step').describe().drop('statistic')
    return df.to_numpy().flatten(), filename.name.removeprefix('id=')


def load_time_series(ids) -> pl.DataFrame:
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    df = pl.DataFrame(stats, schema=[f"stat_{i}" for i in range(len(stats[0]))], orient="row")
    df = df.with_columns(pl.Series('id', indexes))
    
    return df
