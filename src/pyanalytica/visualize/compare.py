"""Comparison visualizations — grouped boxplot, violin, bar of means, strip plot."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyanalytica.core.codegen import CodeSnippet

Figure = matplotlib.figure.Figure


def grouped_boxplot(
    df: pd.DataFrame, x_cat: str, y_num: str,
    sort_by: str = "mean",
) -> tuple[Figure, CodeSnippet]:
    """Create a grouped box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if sort_by == "mean":
        order = df.groupby(x_cat)[y_num].mean().sort_values(ascending=False).index.tolist()
    elif sort_by == "median":
        order = df.groupby(x_cat)[y_num].median().sort_values(ascending=False).index.tolist()
    else:
        order = sorted(df[x_cat].dropna().unique())

    sns.boxplot(data=df, x=x_cat, y=y_num, order=order, ax=ax)
    ax.set_title(f"{y_num} by {x_cat}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(10, 6))\n'
        f'order = df.groupby("{x_cat}")["{y_num}"].mean().sort_values(ascending=False).index\n'
        f'sns.boxplot(data=df, x="{x_cat}", y="{y_num}", order=order, ax=ax)\n'
        f'ax.set_title("{y_num} by {x_cat}")\n'
        f'plt.xticks(rotation=45, ha="right")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def grouped_violin(
    df: pd.DataFrame, x_cat: str, y_num: str,
) -> tuple[Figure, CodeSnippet]:
    """Create a grouped violin plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby(x_cat)[y_num].mean().sort_values(ascending=False).index.tolist()
    sns.violinplot(data=df, x=x_cat, y=y_num, order=order, ax=ax)
    ax.set_title(f"{y_num} by {x_cat}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(10, 6))\n'
        f'sns.violinplot(data=df, x="{x_cat}", y="{y_num}", ax=ax)\n'
        f'ax.set_title("{y_num} by {x_cat}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def bar_of_means(
    df: pd.DataFrame, x_cat: str, y_num: str,
    error_bars: bool = True,
) -> tuple[Figure, CodeSnippet]:
    """Create a bar chart of means with optional error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ci_str = "95" if error_bars else None
    sns.barplot(data=df, x=x_cat, y=y_num, ci=ci_str if error_bars else None,
                errorbar=("ci", 95) if error_bars else None, ax=ax)
    ax.set_title(f"Mean {y_num} by {x_cat}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    err_str = ', errorbar=("ci", 95)' if error_bars else ""
    code = (
        f'fig, ax = plt.subplots(figsize=(10, 6))\n'
        f'sns.barplot(data=df, x="{x_cat}", y="{y_num}"{err_str}, ax=ax)\n'
        f'ax.set_title("Mean {y_num} by {x_cat}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def strip_plot(
    df: pd.DataFrame, x_cat: str, y_num: str,
) -> tuple[Figure, CodeSnippet]:
    """Create a strip plot (jittered dot plot)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(data=df, x=x_cat, y=y_num, alpha=0.5, jitter=True, ax=ax)
    ax.set_title(f"{y_num} by {x_cat}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(10, 6))\n'
        f'sns.stripplot(data=df, x="{x_cat}", y="{y_num}", alpha=0.5, jitter=True, ax=ax)\n'
        f'ax.set_title("{y_num} by {x_cat}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )
