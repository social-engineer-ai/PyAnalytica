"""Distribution visualizations — histogram, boxplot, violin, bar chart."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyanalytica.core.codegen import CodeSnippet

Figure = matplotlib.figure.Figure


def histogram(
    df: pd.DataFrame, col: str, bins: int = 30,
    kde: bool = False, ref_lines: bool = True,
) -> tuple[Figure, CodeSnippet]:
    """Create a histogram of a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax)

    if ref_lines:
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="-.", label=f"Median: {median_val:.2f}")
        ax.legend()

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()

    kde_str = ", kde=True" if kde else ""
    code = (
        f'fig, ax = plt.subplots(figsize=(8, 5))\n'
        f'sns.histplot(df["{col}"].dropna(), bins={bins}{kde_str}, ax=ax)\n'
        f'ax.set_title("Distribution of {col}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def boxplot(df: pd.DataFrame, col: str) -> tuple[Figure, CodeSnippet]:
    """Create a boxplot of a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df[col].dropna(), ax=ax)
    ax.set_title(f"Box Plot of {col}")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(8, 5))\n'
        f'sns.boxplot(x=df["{col}"].dropna(), ax=ax)\n'
        f'ax.set_title("Box Plot of {col}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def violin(df: pd.DataFrame, col: str) -> tuple[Figure, CodeSnippet]:
    """Create a violin plot of a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x=df[col].dropna(), ax=ax)
    ax.set_title(f"Violin Plot of {col}")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(8, 5))\n'
        f'sns.violinplot(x=df["{col}"].dropna(), ax=ax)\n'
        f'ax.set_title("Violin Plot of {col}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt", "import seaborn as sns"],
    )


def bar_chart(
    df: pd.DataFrame, col: str, orientation: str = "vertical",
    sort: bool = True, pct: bool = False,
) -> tuple[Figure, CodeSnippet]:
    """Create a bar chart of a categorical column."""
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df[col].value_counts()
    if sort:
        counts = counts.sort_values(ascending=False)

    if pct:
        counts = counts / counts.sum() * 100
        ylabel = "Percentage"
    else:
        ylabel = "Count"

    if orientation == "horizontal":
        counts.plot.barh(ax=ax)
        ax.set_xlabel(ylabel)
    else:
        counts.plot.bar(ax=ax)
        ax.set_ylabel(ylabel)

    ax.set_title(f"{'Percentage' if pct else 'Count'} of {col}")
    fig.tight_layout()

    sort_str = ".sort_values(ascending=False)" if sort else ""
    pct_str = " / counts.sum() * 100" if pct else ""
    plot_type = "barh" if orientation == "horizontal" else "bar"
    code = (
        f'counts = df["{col}"].value_counts(){sort_str}{pct_str}\n'
        f'fig, ax = plt.subplots(figsize=(8, 5))\n'
        f'counts.plot.{plot_type}(ax=ax)\n'
        f'ax.set_title("{"Percentage" if pct else "Count"} of {col}")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt"],
    )
