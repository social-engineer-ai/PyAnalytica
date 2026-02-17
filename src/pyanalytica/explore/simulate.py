"""Probability distributions, Central Limit Theorem, and Law of Large Numbers simulations."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet

Figure = matplotlib.figure.Figure

# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

DISTRIBUTIONS: dict[str, dict] = {
    "normal": {
        "type": "continuous",
        "params": {"loc": ("Mean", 0.0), "scale": ("Std Dev", 1.0)},
    },
    "binomial": {
        "type": "discrete",
        "params": {"n": ("Trials", 10), "p": ("Probability", 0.5)},
    },
    "poisson": {
        "type": "discrete",
        "params": {"mu": ("Rate", 5.0)},
    },
    "uniform": {
        "type": "continuous",
        "params": {"low": ("Low", 0.0), "high": ("High", 1.0)},
    },
    "exponential": {
        "type": "continuous",
        "params": {"scale": ("Scale (1/lambda)", 1.0)},
    },
}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DistributionResult:
    dist_name: str
    samples: np.ndarray
    summary: pd.DataFrame
    figure: Figure
    prob_result: float | None
    prob_description: str
    code: CodeSnippet


@dataclass
class CLTResult:
    sample_means: np.ndarray
    summary: pd.DataFrame
    figure: Figure
    interpretation: str
    code: CodeSnippet


@dataclass
class LLNResult:
    running_means: np.ndarray
    true_mean: float
    summary: pd.DataFrame
    figure: Figure
    interpretation: str
    code: CodeSnippet


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCIPY_MAP = {
    "normal": "norm",
    "binomial": "binom",
    "poisson": "poisson",
    "uniform": "uniform",
    "exponential": "expon",
}


def _get_dist(dist_name: str, params: dict):
    """Return a frozen scipy distribution."""
    scipy_name = _SCIPY_MAP[dist_name]
    dist_cls = getattr(stats, scipy_name)

    if dist_name == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        return dist_cls(loc=low, scale=high - low)
    elif dist_name == "binomial":
        return dist_cls(n=int(params["n"]), p=params["p"])
    elif dist_name == "poisson":
        return dist_cls(mu=params["mu"])
    else:
        return dist_cls(**params)


def _param_str(dist_name: str, params: dict) -> str:
    """Build human-readable parameter string for code snippets."""
    if dist_name == "uniform":
        return f'loc={params.get("low", 0.0)}, scale={params.get("high", 1.0) - params.get("low", 0.0)}'
    elif dist_name == "binomial":
        return f'n={int(params["n"])}, p={params["p"]}'
    elif dist_name == "poisson":
        return f'mu={params["mu"]}'
    else:
        return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate_distribution(
    dist_name: str,
    params: dict,
    sample_size: int = 1000,
    seed: int | None = None,
    prob_calc: str | None = None,
    prob_value: float | None = None,
    prob_value2: float | None = None,
) -> DistributionResult:
    """Sample from a distribution, visualize, and optionally compute a probability.

    Parameters
    ----------
    dist_name : str
        One of the keys in ``DISTRIBUTIONS``.
    params : dict
        Distribution parameters (e.g. ``{"loc": 0, "scale": 1}``).
    sample_size : int
        Number of random draws.
    seed : int | None
        Random seed for reproducibility.
    prob_calc : str | None
        One of ``"leq"``, ``"geq"``, ``"between"``, ``"quantile"``, or ``None``.
    prob_value, prob_value2 : float | None
        Values used for probability calculations.
    """
    rng = np.random.default_rng(seed)
    dist = _get_dist(dist_name, params)
    samples = dist.rvs(size=sample_size, random_state=rng)
    is_discrete = DISTRIBUTIONS[dist_name]["type"] == "discrete"

    # --- Summary statistics ---
    theo_mean = float(dist.mean())
    theo_std = float(dist.std())
    try:
        theo_skew = float(dist.stats(moments="s"))
    except Exception:
        theo_skew = float("nan")
    try:
        theo_kurt = float(dist.stats(moments="k"))
    except Exception:
        theo_kurt = float("nan")

    from scipy.stats import kurtosis as sp_kurt, skew as sp_skew

    summary = pd.DataFrame({
        "Statistic": ["Mean", "Std Dev", "Skewness", "Kurtosis"],
        "Sample": [
            float(np.mean(samples)),
            float(np.std(samples, ddof=1)),
            float(sp_skew(samples)),
            float(sp_kurt(samples)),
        ],
        "Theoretical": [theo_mean, theo_std, theo_skew, theo_kurt],
    })

    # --- Probability calculation ---
    prob_result: float | None = None
    prob_description = ""
    if prob_calc == "leq" and prob_value is not None:
        prob_result = float(dist.cdf(prob_value))
        prob_description = f"P(X <= {prob_value}) = {prob_result:.4f}"
    elif prob_calc == "geq" and prob_value is not None:
        prob_result = float(1 - dist.cdf(prob_value))
        prob_description = f"P(X >= {prob_value}) = {prob_result:.4f}"
    elif prob_calc == "between" and prob_value is not None and prob_value2 is not None:
        prob_result = float(dist.cdf(prob_value2) - dist.cdf(prob_value))
        prob_description = f"P({prob_value} <= X <= {prob_value2}) = {prob_result:.4f}"
    elif prob_calc == "quantile" and prob_value is not None:
        prob_result = float(dist.ppf(prob_value))
        prob_description = f"Quantile({prob_value}) = {prob_result:.4f}"

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    scipy_name = _SCIPY_MAP[dist_name]

    if is_discrete:
        lo = int(samples.min())
        hi = int(samples.max())
        bins_range = np.arange(lo, hi + 2) - 0.5
        counts, _, _ = ax.hist(samples, bins=bins_range, density=True, alpha=0.6,
                               color="steelblue", edgecolor="white", label="Sample")
        x_vals = np.arange(lo, hi + 1)
        ax.plot(x_vals, dist.pmf(x_vals), "ro-", label="PMF", markersize=5)
    else:
        ax.hist(samples, bins=50, density=True, alpha=0.6,
                color="steelblue", edgecolor="white", label="Sample")
        x_lo = float(dist.ppf(0.001))
        x_hi = float(dist.ppf(0.999))
        x_vals = np.linspace(x_lo, x_hi, 300)
        ax.plot(x_vals, dist.pdf(x_vals), "r-", linewidth=2, label="PDF")

        # Shade probability region
        if prob_calc == "leq" and prob_value is not None:
            x_fill = np.linspace(x_lo, prob_value, 200)
            ax.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color="orange", label=prob_description)
        elif prob_calc == "geq" and prob_value is not None:
            x_fill = np.linspace(prob_value, x_hi, 200)
            ax.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color="orange", label=prob_description)
        elif prob_calc == "between" and prob_value is not None and prob_value2 is not None:
            x_fill = np.linspace(prob_value, prob_value2, 200)
            ax.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color="orange", label=prob_description)

    ax.set_title(f"{dist_name.title()} Distribution (n={sample_size})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout(pad=1.5)

    # --- Code snippet ---
    seed_str = f"\nrng = np.random.default_rng({seed})" if seed is not None else "\nrng = np.random.default_rng()"
    code_text = (
        f"{seed_str}\n"
        f"dist = stats.{scipy_name}({_param_str(dist_name, params)})\n"
        f"samples = dist.rvs(size={sample_size}, random_state=rng)\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.hist(samples, bins=50, density=True, alpha=0.6, label='Sample')\n"
    )
    if is_discrete:
        code_text += f"x = np.arange(samples.min(), samples.max() + 1)\nax.plot(x, dist.pmf(x), 'ro-', label='PMF')\n"
    else:
        code_text += f"x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 300)\nax.plot(x, dist.pdf(x), 'r-', lw=2, label='PDF')\n"
    code_text += (
        f"ax.set_title('{dist_name.title()} Distribution')\n"
        f"ax.legend()\nplt.tight_layout()\nplt.show()"
    )

    snippet = CodeSnippet(
        code=code_text.strip(),
        imports=["import numpy as np", "from scipy import stats", "import matplotlib.pyplot as plt"],
    )

    return DistributionResult(
        dist_name=dist_name,
        samples=samples,
        summary=summary,
        figure=fig,
        prob_result=prob_result,
        prob_description=prob_description,
        code=snippet,
    )


def simulate_clt(
    dist_name: str,
    params: dict,
    sample_size: int = 30,
    num_samples: int = 1000,
    seed: int | None = None,
) -> CLTResult:
    """Demonstrate the Central Limit Theorem.

    Draw *num_samples* samples each of size *sample_size* from the given
    distribution and plot the distribution of sample means.
    """
    rng = np.random.default_rng(seed)
    dist = _get_dist(dist_name, params)
    scipy_name = _SCIPY_MAP[dist_name]

    all_samples = dist.rvs(size=(num_samples, sample_size), random_state=rng)
    sample_means = all_samples.mean(axis=1)

    theo_mean = float(dist.mean())
    theo_se = float(dist.std() / np.sqrt(sample_size))
    obs_mean = float(np.mean(sample_means))
    obs_se = float(np.std(sample_means, ddof=1))

    summary = pd.DataFrame({
        "Statistic": ["Mean", "Std Error"],
        "Theoretical": [theo_mean, theo_se],
        "Observed": [obs_mean, obs_se],
    })

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_means, bins=50, density=True, alpha=0.6,
            color="steelblue", edgecolor="white", label="Sample Means")
    x = np.linspace(sample_means.min(), sample_means.max(), 300)
    ax.plot(x, stats.norm.pdf(x, theo_mean, theo_se), "r-", linewidth=2, label="Normal Approx")
    ax.axvline(theo_mean, color="green", linestyle="--", label=f"True Mean = {theo_mean:.3f}")
    ax.set_title(f"CLT: {dist_name.title()} (n={sample_size}, k={num_samples})")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout(pad=1.5)

    interpretation = (
        f"The Central Limit Theorem states that the sampling distribution of the mean "
        f"approaches a normal distribution as sample size increases, regardless of the "
        f"population distribution.\n\n"
        f"Population: {dist_name.title()}({_param_str(dist_name, params)})\n"
        f"Sample size (n): {sample_size}, Number of samples (k): {num_samples}\n"
        f"Theoretical mean: {theo_mean:.4f}, Observed mean: {obs_mean:.4f}\n"
        f"Theoretical SE: {theo_se:.4f}, Observed SE: {obs_se:.4f}"
    )

    seed_str = f"\nrng = np.random.default_rng({seed})" if seed is not None else "\nrng = np.random.default_rng()"
    code_text = (
        f"{seed_str}\n"
        f"dist = stats.{scipy_name}({_param_str(dist_name, params)})\n"
        f"sample_means = np.array([dist.rvs(size={sample_size}, random_state=rng).mean()\n"
        f"                         for _ in range({num_samples})])\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.hist(sample_means, bins=50, density=True, alpha=0.6, label='Sample Means')\n"
        f"x = np.linspace(sample_means.min(), sample_means.max(), 300)\n"
        f"mu, se = dist.mean(), dist.std() / np.sqrt({sample_size})\n"
        f"ax.plot(x, stats.norm.pdf(x, mu, se), 'r-', lw=2, label='Normal Approx')\n"
        f"ax.set_title('Central Limit Theorem')\n"
        f"ax.legend()\nplt.tight_layout()\nplt.show()"
    )

    snippet = CodeSnippet(
        code=code_text.strip(),
        imports=["import numpy as np", "from scipy import stats", "import matplotlib.pyplot as plt"],
    )

    return CLTResult(
        sample_means=sample_means,
        summary=summary,
        figure=fig,
        interpretation=interpretation,
        code=snippet,
    )


def simulate_lln(
    dist_name: str,
    params: dict,
    max_obs: int = 5000,
    seed: int | None = None,
) -> LLNResult:
    """Demonstrate the Law of Large Numbers.

    Plot the running cumulative mean as observations increase.
    """
    rng = np.random.default_rng(seed)
    dist = _get_dist(dist_name, params)
    scipy_name = _SCIPY_MAP[dist_name]

    samples = dist.rvs(size=max_obs, random_state=rng)
    running_means = np.cumsum(samples) / np.arange(1, max_obs + 1)
    true_mean = float(dist.mean())

    final_mean = float(running_means[-1])
    abs_diff = abs(final_mean - true_mean)

    summary = pd.DataFrame({
        "Statistic": ["True Mean", "Final Running Mean", "Absolute Difference", "Observations"],
        "Value": [true_mean, final_mean, abs_diff, max_obs],
    })

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(running_means, color="steelblue", linewidth=0.8, label="Running Mean")
    ax.axhline(true_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"True Mean = {true_mean:.3f}")
    ax.set_title(f"LLN: {dist_name.title()} (n={max_obs})")
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Running Mean")
    ax.legend()
    fig.tight_layout(pad=1.5)

    interpretation = (
        f"The Law of Large Numbers states that as the number of observations increases, "
        f"the sample mean converges to the population mean.\n\n"
        f"Population: {dist_name.title()}({_param_str(dist_name, params)})\n"
        f"True mean: {true_mean:.4f}\n"
        f"Running mean after {max_obs} observations: {final_mean:.4f}\n"
        f"Absolute difference: {abs_diff:.4f}"
    )

    seed_str = f"\nrng = np.random.default_rng({seed})" if seed is not None else "\nrng = np.random.default_rng()"
    code_text = (
        f"{seed_str}\n"
        f"dist = stats.{scipy_name}({_param_str(dist_name, params)})\n"
        f"samples = dist.rvs(size={max_obs}, random_state=rng)\n"
        f"running_means = np.cumsum(samples) / np.arange(1, {max_obs} + 1)\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.plot(running_means, lw=0.8, label='Running Mean')\n"
        f"ax.axhline(dist.mean(), color='red', ls='--', label='True Mean')\n"
        f"ax.set_title('Law of Large Numbers')\n"
        f"ax.legend()\nplt.tight_layout()\nplt.show()"
    )

    snippet = CodeSnippet(
        code=code_text.strip(),
        imports=["import numpy as np", "from scipy import stats", "import matplotlib.pyplot as plt"],
    )

    return LLNResult(
        running_means=running_means,
        true_mean=true_mean,
        summary=summary,
        figure=fig,
        interpretation=interpretation,
        code=snippet,
    )
