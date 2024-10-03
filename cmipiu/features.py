"""
This module is responsible for all functions for feature engineering
"""

import polars as pl
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
