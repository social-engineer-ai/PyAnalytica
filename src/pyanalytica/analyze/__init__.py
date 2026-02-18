"""PyAnalytica analyze module."""

from pyanalytica.analyze.correlation import correlation_test
from pyanalytica.analyze.means import (
    kruskal_wallis_test, mann_whitney_test, one_sample_ttest, one_way_anova, two_sample_ttest,
)
from pyanalytica.analyze.normality import NormalityResult, shapiro_wilk_test
from pyanalytica.analyze.proportions import (
    GoodnessOfFitResult, OnePropResult, TwoPropResult,
    chi_square_test, goodness_of_fit_test, one_proportion_ztest, two_proportion_ztest,
)

__all__ = [
    "correlation_test",
    "one_sample_ttest",
    "two_sample_ttest",
    "one_way_anova",
    "mann_whitney_test",
    "kruskal_wallis_test",
    "shapiro_wilk_test",
    "NormalityResult",
    "chi_square_test",
    "goodness_of_fit_test",
    "GoodnessOfFitResult",
    "one_proportion_ztest",
    "two_proportion_ztest",
    "OnePropResult",
    "TwoPropResult",
]
