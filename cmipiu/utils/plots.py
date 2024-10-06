"""
Contains utility functions for visualization
"""

import polars as pl
import altair as alt

def viz_crosstab(data: pl.DataFrame, x: str, y: str, normalize_over_x: bool = False, width: int = None, height: int = None, scheme: str = "lighttealblue"):
    group_counts = data.group_by([x, y]).agg(
        count=pl.col(x).count()
    )

    if normalize_over_x:
        group_counts = group_counts.with_columns(
            (pl.col('count') / pl.col('count').sum()).over(x).round(2)
        )
    
    minv = group_counts.select('count').min().item()
    maxv = group_counts.select('count').max().item()
    value_range = minv + (maxv - minv) * 0.7

    n1 = group_counts.select(x).n_unique()
    n2 = group_counts.select(y).n_unique()
    if width is None: width = 100+30*n1
    if height is None: height = 50+30*n2

    heatmap = alt.Chart(group_counts).mark_rect(stroke='black', strokeWidth=1).encode(
        alt.X(field=x, type='nominal'),
        alt.Y(field=y, type='ordinal', scale=alt.Scale()),
        alt.Color('count', scale=alt.Scale(scheme=scheme)),
    )

    text = heatmap.mark_text().encode(
        alt.X(field=x, type='nominal'),
        alt.Y(field=y, type='ordinal'),
        alt.Text(field='count', type='quantitative'),
        color=alt.condition(
            alt.datum.count >= value_range,
            alt.value('white'),
            alt.value('black')
        )
    ).properties(
        width=width,
        height=height,
    )

    return heatmap + text
