"""Time series visualizations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyanalytica.core.codegen import CodeSnippet

Figure = matplotlib.figure.Figure


def time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_by: str | None = None,
    agg_level: str = "raw",
    chart_type: str = "line",
    rolling_window: int | None = None,
) -> tuple[Figure, CodeSnippet]:
    """Create a time series chart.

    agg_level: 'raw', 'daily', 'weekly', 'monthly'
    chart_type: 'line', 'area', 'bar'
    """
    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    work_df = work_df.sort_values(date_col)

    code_lines = [
        f'df["{date_col}"] = pd.to_datetime(df["{date_col}"])',
        f'df = df.sort_values("{date_col}")',
    ]

    # Aggregate if needed
    freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
    if agg_level in freq_map:
        freq = freq_map[agg_level]
        if group_by:
            work_df = work_df.set_index(date_col).groupby(group_by)[value_col].resample(freq).mean().reset_index()
            code_lines.append(
                f'df = df.set_index("{date_col}").groupby("{group_by}")["{value_col}"]'
                f'.resample("{freq}").mean().reset_index()'
            )
        else:
            work_df = work_df.set_index(date_col)[value_col].resample(freq).mean().reset_index()
            code_lines.append(
                f'df = df.set_index("{date_col}")["{value_col}"]'
                f'.resample("{freq}").mean().reset_index()'
            )

    fig, ax = plt.subplots(figsize=(12, 5))

    if group_by and group_by in work_df.columns:
        for name, group in work_df.groupby(group_by):
            if chart_type == "line":
                ax.plot(group[date_col], group[value_col], label=str(name))
            elif chart_type == "area":
                ax.fill_between(group[date_col], group[value_col], alpha=0.5, label=str(name))
            elif chart_type == "bar":
                ax.bar(group[date_col], group[value_col], alpha=0.7, label=str(name))
        ax.legend()
    else:
        if chart_type == "line":
            ax.plot(work_df[date_col], work_df[value_col])
        elif chart_type == "area":
            ax.fill_between(work_df[date_col], work_df[value_col], alpha=0.5)
        elif chart_type == "bar":
            ax.bar(work_df[date_col], work_df[value_col])

    code_lines.append(f'fig, ax = plt.subplots(figsize=(12, 5))')
    code_lines.append(f'ax.plot(df["{date_col}"], df["{value_col}"])')

    # Rolling average
    if rolling_window and not group_by:
        rolling = work_df.set_index(date_col)[value_col].rolling(rolling_window).mean()
        ax.plot(rolling.index, rolling.values, "r-", linewidth=2,
                label=f"{rolling_window}-period rolling avg")
        ax.legend()
        code_lines.append(
            f'rolling = df.set_index("{date_col}")["{value_col}"].rolling({rolling_window}).mean()\n'
            f'ax.plot(rolling.index, rolling.values, "r-", linewidth=2, '
            f'label="{rolling_window}-period rolling avg")'
        )

    ax.set_title(f"{value_col} over Time")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    code_lines.extend([
        f'ax.set_title("{value_col} over Time")',
        f'plt.xticks(rotation=45, ha="right")',
        f'plt.tight_layout()',
        f'plt.show()',
    ])

    return fig, CodeSnippet(
        code="\n".join(code_lines),
        imports=["import matplotlib.pyplot as plt", "import pandas as pd"],
    )
