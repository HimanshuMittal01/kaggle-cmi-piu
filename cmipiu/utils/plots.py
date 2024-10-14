"""
Contains utility functions for visualization
"""

import polars as pl
import altair as alt

def viz_crosstab(data: pl.DataFrame, x: str, y: str, normalize_over_x: bool = False, width: int = None, height: int = None, scheme: str = "lighttealblue", precision: int = 0):
    group_counts = data.group_by([x, y]).agg(
        count=pl.col(x).len()
    )

    if normalize_over_x:
        if not precision: precision=2
        group_counts = group_counts.with_columns(
            (pl.col('count') / pl.col('count').sum()).over(x)
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
        alt.Y(field=y, type='ordinal'),
        alt.Color('count', scale=alt.Scale(scheme=scheme)),
    )

    text = heatmap.mark_text().encode(
        alt.X(field=x, type='nominal'),
        alt.Y(field=y, type='ordinal'),
        alt.Text(field='count', type='quantitative', format=f'0.{precision}f'),
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


def viz_piecounts(data: pl.DataFrame, x: str, width: int = 0, height: int = 0, scheme: str = "category10", colorreverse: bool = True):
    counts = data[x].value_counts().with_columns(
        percentage=pl.col('count') / pl.col('count').sum()
    )

    base = alt.Chart(counts).encode(
        alt.Theta('count:Q', stack=True),
    )

    pie = base.mark_arc(outerRadius=120, padAngle=0.03, innerRadius=60).encode(
        alt.Color(
            field=x,
            type='nominal',
            scale=alt.Scale(scheme=scheme, reverse=colorreverse),
            legend=alt.Legend(orient='bottom-right', offset=0, title=None, direction='horizontal')
        ),
        alt.Order("percentage", sort="descending"),
    )
    text1 = base.mark_text(radius=140, fontWeight='bold').encode(
        alt.Text('percentage:Q', format='.1%'),
        alt.Order("percentage", sort="descending"),
        color=alt.value('black')
    )

    text2 = base.mark_text(radius=90).encode(
        alt.Text('count:Q'),
        alt.Order("percentage", sort="descending"),
        color=alt.value('black')
    )

    chart = (pie + text1 + text2).properties(
        title=x,
        width=width,
        height=height
    )

    return chart


def viz_countplot(data: pl.DataFrame, x: str, barcolor: str = '#1f77b4', enableText: bool = True):
    counts = data[x].value_counts()
    bar = alt.Chart(counts).mark_bar(color=barcolor).encode(
        alt.X(field=x, type='ordinal'),
        alt.Y(field='count', type='quantitative'),
    )

    if enableText:
        text = bar.mark_text().transform_calculate(
            textPos=alt.datum.count + 0.025 * counts['count'].max()
        ).encode(
            alt.Y('textPos:Q', axis=alt.Axis(title='count')),
            alt.Text('count:Q')
        )
        return bar + text
    
    return bar


def viz_areaplot(data: pl.DataFrame, x: str, width: int = 0, height: int = 0, barcolor: str = '#1f77b4'):
    chart = alt.Chart(data).mark_area(color=barcolor).encode(
        alt.X(field=x, type="quantitative"),
        alt.Y(field=x, aggregate='count', type="quantitative"),
    ).properties(
        width=width,
        height=height,
    )

    return chart


def viz_histogram(data: pl.DataFrame, x: str, width: int = 0, height: int = 0, barcolor: str = '#1f77b4', bin: alt.Bin = None):
    chart = alt.Chart(data).mark_bar(color=barcolor).encode(
        alt.X(field=x, type='quantitative', bin=bin),
        alt.Y('count()')
    ).properties(
        width=width,
        height=height
    )

    return chart


def viz_bubble_chart(data: pl.DataFrame, x: str, y: str, normalize_over_x: bool = False, width: int = None, height: int = None, scheme: str = "lighttealblue"):
    group_counts = data.group_by([x, y]).agg(
        count=pl.col(x).len()
    )

    if normalize_over_x:
        group_counts = group_counts.with_columns(
            (pl.col('count') / pl.col('count').sum()).over(x)
        )

    n1 = group_counts.select(x).n_unique()
    n2 = group_counts.select(y).n_unique()
    if width is None: width = 100+20*n1
    if height is None: height = 50+20*n2

    heatmap = alt.Chart(group_counts).mark_circle(stroke='black', strokeWidth=1).encode(
        alt.X(field=x, type='ordinal'),
        alt.Y(field=y, type='ordinal'),
        alt.Color('count', scale=alt.Scale(scheme=scheme)),
        alt.Size('count'),
    ).properties(
        width=width,
        height=height,
    )

    return heatmap


def viz_kdeplot(data: pl.DataFrame, x: str, color: str = None, width: int = 0, height: int = 0, scheme: str = "lighttealblue", cumulative: bool = False):
    if color is not None:
        chart = alt.Chart(data).transform_density(
            groupby=[color],
            density=x,
            cumulative=cumulative,
            as_=[x, 'density']
        ).mark_line().encode(
            alt.X(field=x, type="quantitative"),
            alt.Y('density:Q'),
            alt.Color(field=color, type='nominal', scale=alt.Scale(scheme=scheme))
        )
    
    else:
        chart = alt.Chart(data).transform_density(
            density=x,
            cumulative=cumulative,
            as_=[x, 'density']
        ).mark_line().encode(
            alt.X(field=x, type="quantitative"),
            alt.Y('density:Q'),
        )
    
    chart = chart.properties(
        width=width,
        height=height,
    )

    return chart


def viz_violin_boxplot(data: pl.DataFrame, x: str, y: str, scheme: str = "category10", boxcolor: str = '#3e3e3e'):
    base = alt.Chart(data, width=100).encode(
        alt.Y(field=y, type='quantitative'),
    )

    chart1 = base.transform_density(
        y,
        as_=[y, 'density'],
        extent=[data[y].min(), data[y].max()],
        groupby=[x]
    ).mark_area(orient='horizontal').encode(
        alt.Color(field=x, type='nominal', scale=alt.Scale(scheme=scheme)),
        alt.X('density:Q')
            .stack('center')
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0], grid=False, ticks=True),
    )

    chart2 = base.mark_boxplot(color=boxcolor)

    final_chart = alt.layer(chart1, chart2).facet(
        f'{x}:N'
    ).resolve_scale(
        x='independent'
    )

    return final_chart


def viz_corrheatmap(data: pl.DataFrame, width: int = 0, height: int = 0, annot: bool = False, precision: int = 2, method: str = 'pearson'):
    _new_data = []
    for col1 in data.columns:
        for col2 in data.columns:
            _new_data.append([col1, col2, data.drop_nulls(subset=[col1, col2]).select(pl.corr(pl.col(col1), pl.col(col2), method=method)).item()])

    df = pl.DataFrame(_new_data, schema=['column1', 'column2', 'value'], orient='row')

    base = alt.Chart(df).encode(
        alt.X(field='column1', type='nominal', title=None),
        alt.Y(field='column2', type='nominal', title=None),
    )

    heatmap = base.mark_rect().encode(
        alt.Color('value:Q')
    )

    if annot:
        text = base.mark_text().encode(
            alt.Text(field='value', type='quantitative', format=f'0.{precision}f'),
            color=alt.condition(
                alt.datum.value >= 0.7,
                alt.value('white'),
                alt.value('black')
            )
        ).properties(
            width=width,
            height=height,
        )

        return heatmap + text

    return heatmap
