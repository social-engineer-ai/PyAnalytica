"""Relationship visualizations — scatter, hexbin with trend lines."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet

Figure = matplotlib.figure.Figure


def scatter(
    df: pd.DataFrame, x: str, y: str,
    color_by: str | None = None, size_by: str | None = None,
    trend_line: bool = True,
) -> tuple[Figure, CodeSnippet]:
    """Create a scatter plot of two numeric variables."""
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_kwargs: dict = {"alpha": 0.6}
    if color_by and color_by in df.columns:
        plot_kwargs["hue"] = color_by
    if size_by and size_by in df.columns:
        plot_kwargs["size"] = size_by

    clean = df[[x, y]].dropna()

    sns.scatterplot(data=df, x=x, y=y, ax=ax, **plot_kwargs)

    code_lines = [
        f'fig, ax = plt.subplots(figsize=(8, 6))',
        f'sns.scatterplot(data=df, x="{x}", y="{y}", alpha=0.6, ax=ax)',
    ]

    if trend_line and len(clean) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean[x], clean[y])
        x_range = np.linspace(clean[x].min(), clean[x].max(), 100)
        ax.plot(x_range, intercept + slope * x_range, "r--", linewidth=2,
                label=f"R\u00b2 = {r_value**2:.3f}")
        ax.legend()

        code_lines.extend([
            f'from scipy import stats',
            f'slope, intercept, r, p, se = stats.linregress(df["{x}"].dropna(), df["{y}"].dropna())',
            f'x_range = np.linspace(df["{x}"].min(), df["{x}"].max(), 100)',
            f'ax.plot(x_range, intercept + slope * x_range, "r--", label=f"R\u00b2 = {{r**2:.3f}}")',
            f'ax.legend()',
        ])

    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()

    code_lines.extend([
        f'ax.set_title("{y} vs {x}")',
        f'plt.tight_layout()',
        f'plt.show()',
    ])

    return fig, CodeSnippet(
        code="\n".join(code_lines),
        imports=["import matplotlib.pyplot as plt", "import numpy as np", "import seaborn as sns"],
    )


def hexbin(
    df: pd.DataFrame, x: str, y: str, gridsize: int = 30,
) -> tuple[Figure, CodeSnippet]:
    """Create a hexbin plot for large datasets."""
    fig, ax = plt.subplots(figsize=(8, 6))
    clean = df[[x, y]].dropna()
    hb = ax.hexbin(clean[x], clean[y], gridsize=gridsize, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} (hexbin)")
    fig.tight_layout()

    code = (
        f'fig, ax = plt.subplots(figsize=(8, 6))\n'
        f'hb = ax.hexbin(df["{x}"], df["{y}"], gridsize={gridsize}, cmap="YlOrRd", mincnt=1)\n'
        f'fig.colorbar(hb, ax=ax, label="Count")\n'
        f'ax.set_title("{y} vs {x} (hexbin)")\n'
        f'plt.tight_layout()\n'
        f'plt.show()'
    )

    return fig, CodeSnippet(
        code=code,
        imports=["import matplotlib.pyplot as plt"],
    )
