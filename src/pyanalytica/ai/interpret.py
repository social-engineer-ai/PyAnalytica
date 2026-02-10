"""AI-powered interpretation of statistical and modeling results.

Provides two modes:
1. Rule-based (always available): Pattern-matches on known result types
   to generate clear, educational plain-English explanations.
2. LLM-enhanced (optional): When an Anthropic API key is set, sends
   the rule-based interpretation to Claude for richer, more nuanced output.

The rule-based engine covers every result type produced by pyanalytica's
analyze and model packages, including means tests, correlations,
proportions tests, regressions, classifications, clustering, and PCA.

Usage:
    from pyanalytica.ai.interpret import interpret_result
    explanation = interpret_result(my_result, context="marketing survey data")
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# LLM helper (optional -- never crashes if anthropic is missing)
# ---------------------------------------------------------------------------

def _try_llm(prompt: str) -> str | None:
    """Attempt to get an LLM-enhanced response from Claude.

    Returns the response text on success, or ``None`` if the anthropic
    package is not installed or the API key is not configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        # Any failure (missing package, network error, bad key) -- fall back
        return None


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    """Format a p-value for display."""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def _effect_size_label(value: float, name: str) -> str:
    """Return a human-readable label for common effect-size metrics."""
    name_lower = name.lower()
    if "cohen" in name_lower or name_lower == "d":
        if value < 0.2:
            return "negligible"
        elif value < 0.5:
            return "small"
        elif value < 0.8:
            return "medium"
        else:
            return "large"
    if "eta" in name_lower:
        if value < 0.01:
            return "negligible"
        elif value < 0.06:
            return "small"
        elif value < 0.14:
            return "medium"
        else:
            return "large"
    if name_lower in ("r", "r_squared", "r-squared"):
        if value < 0.04:
            return "negligible"
        elif value < 0.25:
            return "small"
        elif value < 0.64:
            return "medium"
        else:
            return "large"
    return ""


def _significance_statement(p_value: float, alpha: float = 0.05) -> str:
    """Return a plain-English significance statement."""
    if p_value < alpha:
        return (
            f"The p-value ({_fmt_p(p_value)}) is below the conventional "
            f"significance level of {alpha}, so we reject the null hypothesis."
        )
    return (
        f"The p-value ({_fmt_p(p_value)}) is above the conventional "
        f"significance level of {alpha}, so we fail to reject the null "
        f"hypothesis. This does not prove there is no effect -- it means we "
        f"do not have enough evidence to conclude one exists."
    )


# ---------------------------------------------------------------------------
# Result-type interpreters (rule-based)
# ---------------------------------------------------------------------------

def _interpret_means_test(result: Any, context: str) -> str:
    """Interpret a MeansTestResult from pyanalytica.analyze.means."""
    lines: list[str] = []

    test_name = getattr(result, "test_name", "Means test")
    lines.append(f"TEST: {test_name}")
    lines.append("")

    # What the test does
    if "one-sample" in test_name.lower():
        lines.append(
            "This test compares the average of a single sample to a "
            "specific hypothesized value.  It asks: 'Is the true population "
            "mean different from the value we expected?'"
        )
    elif "two-sample" in test_name.lower() or "welch" in test_name.lower():
        lines.append(
            "This test compares the averages of two independent groups to "
            "see if they are statistically different.  It asks: 'Do these "
            "two groups come from populations with the same mean?'"
        )
    elif "anova" in test_name.lower():
        lines.append(
            "ANOVA (Analysis of Variance) compares the averages of three "
            "or more groups simultaneously.  It asks: 'Is at least one "
            "group mean different from the others?'"
        )
    else:
        lines.append(
            f"This is a {test_name} -- it evaluates whether observed "
            f"differences in group means are statistically meaningful."
        )
    lines.append("")

    # Test statistic and p-value
    stat = getattr(result, "statistic", None)
    p_value = getattr(result, "p_value", None)
    if stat is not None and p_value is not None:
        lines.append(f"Test statistic = {stat:.4f}, p-value = {_fmt_p(p_value)}")
        lines.append("")
        lines.append(_significance_statement(p_value))
        lines.append("")

    # Effect size
    es = getattr(result, "effect_size", None)
    es_name = getattr(result, "effect_size_name", "")
    if es is not None and es_name:
        label = _effect_size_label(es, es_name)
        label_str = f" ({label})" if label else ""
        lines.append(
            f"Effect size ({es_name}) = {es:.4f}{label_str}. "
            f"While the p-value tells you whether an effect exists, the "
            f"effect size tells you how large that effect is in practical terms."
        )
        lines.append("")

    # Confidence interval
    ci = getattr(result, "confidence_interval", None)
    if ci is not None and ci != (None, None):
        lines.append(
            f"95% Confidence interval: [{ci[0]:.4f}, {ci[1]:.4f}]. "
            f"We are 95% confident the true population mean falls within "
            f"this range."
        )
        lines.append("")

    # Assumption checks
    checks = getattr(result, "assumption_checks", {})
    if checks:
        lines.append("ASSUMPTION CHECKS:")
        normality_ok = checks.get("normality_ok")
        if normality_ok is not None:
            status = "met" if normality_ok else "potentially violated"
            lines.append(
                f"  - Normality assumption: {status}. "
                f"(Shapiro-Wilk p = {checks.get('normality_shapiro_p', 'N/A')})"
            )
        eq_var = checks.get("equal_variance")
        if eq_var is not None:
            status = "met" if eq_var else "potentially violated"
            lines.append(
                f"  - Equal variance assumption: {status}. "
                f"(Levene's test p = {checks.get('levene_p', 'N/A')})"
            )
        lines.append("")

    if context:
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_correlation(result: Any, context: str) -> str:
    """Interpret a CorrelationResult from pyanalytica.analyze.correlation."""
    lines: list[str] = []

    method = getattr(result, "method", "pearson")
    method_label = "Pearson's r" if method == "pearson" else "Spearman's rho"
    lines.append(f"TEST: {method_label} Correlation")
    lines.append("")

    if method == "pearson":
        lines.append(
            "Pearson's correlation measures the strength and direction of "
            "the LINEAR relationship between two continuous variables.  "
            "Values range from -1 (perfect negative) to +1 (perfect positive), "
            "with 0 meaning no linear relationship."
        )
    else:
        lines.append(
            "Spearman's rank correlation measures the strength and direction "
            "of the MONOTONIC relationship between two variables.  It works "
            "on ranks rather than raw values, making it more robust to "
            "outliers and non-linear but still monotonic relationships."
        )
    lines.append("")

    r = getattr(result, "r", None)
    p_value = getattr(result, "p_value", None)
    n = getattr(result, "n", None)

    if r is not None:
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        direction = "positive" if r > 0 else "negative"
        lines.append(
            f"Correlation coefficient (r) = {r:.4f}. "
            f"This indicates a {strength} {direction} relationship."
        )
        if direction == "positive":
            lines.append(
                "  -> As one variable increases, the other tends to increase as well."
            )
        else:
            lines.append(
                "  -> As one variable increases, the other tends to decrease."
            )
        lines.append("")

    if p_value is not None:
        lines.append(_significance_statement(p_value))
        lines.append("")

    ci_lo = getattr(result, "ci_lower", None)
    ci_hi = getattr(result, "ci_upper", None)
    if ci_lo is not None and ci_hi is not None:
        lines.append(
            f"95% Confidence interval for r: [{ci_lo:.4f}, {ci_hi:.4f}]."
        )
        lines.append("")

    if n is not None:
        lines.append(f"Sample size: n = {n}.")
        lines.append("")

    lines.append(
        "IMPORTANT: Correlation does NOT imply causation.  Even a very "
        "strong correlation could be driven by a confounding variable, "
        "reverse causality, or coincidence."
    )

    if context:
        lines.append("")
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_proportions(result: Any, context: str) -> str:
    """Interpret a ProportionsResult from pyanalytica.analyze.proportions."""
    lines: list[str] = []

    lines.append("TEST: Chi-Square Test of Independence")
    lines.append("")
    lines.append(
        "The chi-square test checks whether two CATEGORICAL variables are "
        "associated (related) or independent.  It compares the observed "
        "counts in each cell of a contingency table to what we would expect "
        "if the variables were completely unrelated."
    )
    lines.append("")

    chi2 = getattr(result, "chi2", None)
    p_value = getattr(result, "p_value", None)
    dof = getattr(result, "dof", None)

    if chi2 is not None and p_value is not None and dof is not None:
        lines.append(
            f"Chi-square statistic = {chi2:.2f}, degrees of freedom = {dof}, "
            f"p-value = {_fmt_p(p_value)}"
        )
        lines.append("")
        lines.append(_significance_statement(p_value))
        if p_value < 0.05:
            lines.append(
                "This means there IS a statistically significant association "
                "between the two variables -- they are not independent."
            )
        else:
            lines.append(
                "This means we do NOT have sufficient evidence of an "
                "association between the two variables."
            )
        lines.append("")

    # Residuals guidance
    lines.append(
        "READING THE RESIDUALS TABLE: Standardized residuals show which "
        "cells deviate most from what we would expect.  Values above +2 "
        "or below -2 indicate cells where the observed count is notably "
        "higher or lower than expected, respectively."
    )

    if context:
        lines.append("")
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_regression(result: Any, context: str) -> str:
    """Interpret a RegressionResult from pyanalytica.model.regression."""
    lines: list[str] = []

    lines.append("MODEL: Linear Regression")
    lines.append("")
    lines.append(
        "Linear regression models the relationship between a continuous "
        "outcome (dependent variable) and one or more predictors "
        "(independent variables).  It finds the best-fitting straight "
        "line (or plane, with multiple predictors) through the data."
    )
    lines.append("")

    r_sq = getattr(result, "r_squared", None)
    adj_r_sq = getattr(result, "adj_r_squared", None)
    f_stat = getattr(result, "f_stat", None)
    f_pvalue = getattr(result, "f_pvalue", None)

    if r_sq is not None:
        pct = r_sq * 100
        label = _effect_size_label(r_sq, "r_squared")
        label_str = f" ({label} effect)" if label else ""
        lines.append(
            f"R-squared = {r_sq:.4f}{label_str}. This means the model "
            f"explains about {pct:.1f}% of the variation in the outcome. "
            f"The remaining {100 - pct:.1f}% is unexplained."
        )
    if adj_r_sq is not None:
        lines.append(
            f"Adjusted R-squared = {adj_r_sq:.4f}. This version penalizes "
            f"for adding extra predictors that don't improve the model, "
            f"making it better for comparing models with different numbers "
            f"of predictors."
        )
    lines.append("")

    if f_stat is not None and f_pvalue is not None:
        lines.append(
            f"F-statistic = {f_stat:.4f}, p-value = {_fmt_p(f_pvalue)}. "
            f"This tests whether the overall model is statistically "
            f"significant (i.e., at least one predictor matters)."
        )
        lines.append("")

    # Coefficient interpretation
    coef_df = getattr(result, "coefficients", None)
    if coef_df is not None and hasattr(coef_df, "iterrows"):
        sig_predictors = []
        nonsig_predictors = []
        for _, row in coef_df.iterrows():
            var_name = row.get("variable", "")
            if var_name == "(Intercept)":
                continue
            p_val = row.get("p_value", 1.0)
            coef = row.get("coefficient", 0.0)
            if p_val < 0.05:
                direction = "increase" if coef > 0 else "decrease"
                sig_predictors.append(
                    f"  - {var_name}: coefficient = {coef:.4f} (p = "
                    f"{_fmt_p(p_val)}). A one-unit increase in {var_name} "
                    f"is associated with a {abs(coef):.4f} {direction} "
                    f"in the outcome, holding other predictors constant."
                )
            else:
                nonsig_predictors.append(var_name)

        if sig_predictors:
            lines.append("SIGNIFICANT PREDICTORS (p < 0.05):")
            lines.extend(sig_predictors)
            lines.append("")
        if nonsig_predictors:
            lines.append(
                f"Non-significant predictors: {', '.join(nonsig_predictors)}. "
                f"These do not have a statistically significant relationship "
                f"with the outcome in this model."
            )
            lines.append("")

    # VIF
    vif_df = getattr(result, "vif", None)
    if vif_df is not None and hasattr(vif_df, "iterrows"):
        high_vif = []
        for _, row in vif_df.iterrows():
            vif_val = row.get("VIF", 0)
            if vif_val > 5:
                high_vif.append(f"{row.get('variable', '?')} (VIF = {vif_val:.1f})")
        if high_vif:
            lines.append(
                f"WARNING -- Multicollinearity detected: {', '.join(high_vif)}. "
                f"VIF > 5 suggests these predictors are highly correlated with "
                f"each other, which can make individual coefficient estimates "
                f"unreliable."
            )
            lines.append("")

    if context:
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_classification(result: Any, context: str) -> str:
    """Interpret a ClassificationResult from pyanalytica.model.classify."""
    lines: list[str] = []

    model_type = getattr(result, "model_type", "Classification")
    lines.append(f"MODEL: {model_type}")
    lines.append("")

    if "logistic" in model_type.lower():
        lines.append(
            "Logistic regression predicts the probability that an "
            "observation belongs to a particular category.  Despite its "
            "name, it is a classification algorithm, not a regression."
        )
    elif "tree" in model_type.lower():
        lines.append(
            "A decision tree classifier makes predictions by learning "
            "simple if-then rules from the data.  It is easy to interpret "
            "and visualize, but can overfit if not properly constrained."
        )
    else:
        lines.append(f"This is a {model_type} classification model.")
    lines.append("")

    train_acc = getattr(result, "train_accuracy", None)
    test_acc = getattr(result, "test_accuracy", None)

    if train_acc is not None:
        lines.append(f"Training accuracy: {train_acc:.1%}")
    if test_acc is not None:
        lines.append(f"Test accuracy: {test_acc:.1%}")
        lines.append(
            "Test accuracy tells us how well the model generalizes to "
            "data it has never seen before -- this is the more important "
            "number."
        )
    lines.append("")

    if train_acc is not None and test_acc is not None:
        gap = train_acc - test_acc
        if gap > 0.10:
            lines.append(
                f"WARNING -- Possible overfitting: the training accuracy "
                f"({train_acc:.1%}) is notably higher than the test accuracy "
                f"({test_acc:.1%}), a gap of {gap:.1%}. The model may be "
                f"memorizing noise in the training data rather than learning "
                f"general patterns."
            )
            lines.append("")
        elif gap < -0.02:
            lines.append(
                "NOTE: Test accuracy is slightly higher than training "
                "accuracy. This can happen with small datasets or lucky "
                "test splits and is not usually concerning."
            )
            lines.append("")

    # Feature importance (decision tree)
    fi_df = getattr(result, "feature_importance", None)
    if fi_df is not None and hasattr(fi_df, "iterrows"):
        lines.append("FEATURE IMPORTANCE (most important first):")
        for _, row in fi_df.iterrows():
            var = row.get("variable", "?")
            imp = row.get("importance", 0)
            bar = "#" * int(imp * 30)
            lines.append(f"  - {var}: {imp:.4f} {bar}")
        lines.append(
            "Feature importance shows how much each variable contributed "
            "to the model's decisions."
        )
        lines.append("")

    # Coefficients (logistic regression)
    coef_df = getattr(result, "coefficients", None)
    if coef_df is not None and fi_df is None and hasattr(coef_df, "iterrows"):
        lines.append("COEFFICIENTS:")
        for _, row in coef_df.iterrows():
            var = row.get("variable", "?")
            coef = row.get("coefficient", 0)
            odds = row.get("odds_ratio", None)
            odds_str = f", odds ratio = {odds:.4f}" if odds is not None else ""
            lines.append(f"  - {var}: coefficient = {coef:.4f}{odds_str}")
        if "odds_ratio" in (coef_df.columns if hasattr(coef_df, "columns") else []):
            lines.append(
                "An odds ratio > 1 means the predictor increases the odds "
                "of the positive outcome; < 1 means it decreases the odds."
            )
        lines.append("")

    classes = getattr(result, "classes", [])
    if classes:
        lines.append(f"Classes predicted: {classes}")

    if context:
        lines.append("")
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_cluster(result: Any, context: str) -> str:
    """Interpret a ClusterResult from pyanalytica.model.cluster."""
    lines: list[str] = []

    n_clusters = getattr(result, "n_clusters", "?")
    lines.append(f"MODEL: Clustering (k = {n_clusters})")
    lines.append("")
    lines.append(
        "Clustering is an UNSUPERVISED technique -- it groups similar "
        "observations together without being told the 'right answer.'  "
        "The goal is to discover natural groupings in the data."
    )
    lines.append("")

    # Silhouette score
    sil_scores = getattr(result, "silhouette_scores", [])
    if sil_scores:
        if len(sil_scores) == 1:
            best_sil = sil_scores[0]
        else:
            best_sil = max(sil_scores)
        lines.append(f"Silhouette score: {best_sil:.4f}")
        if best_sil > 0.7:
            quality = "strong"
        elif best_sil > 0.5:
            quality = "reasonable"
        elif best_sil > 0.25:
            quality = "weak"
        else:
            quality = "poor"
        lines.append(
            f"Cluster quality: {quality}. The silhouette score ranges "
            f"from -1 to +1.  Higher values mean observations are well "
            f"matched to their own cluster and poorly matched to neighboring "
            f"clusters."
        )
        lines.append("")

    # Cluster profiles
    profiles = getattr(result, "cluster_profiles", None)
    if profiles is not None and hasattr(profiles, "to_string") and len(profiles) > 0:
        lines.append("CLUSTER PROFILES (mean values per cluster):")
        lines.append(profiles.to_string())
        lines.append("")
        lines.append(
            "Look for variables where cluster means differ substantially -- "
            "these are the features that distinguish one cluster from another. "
            "Try to give each cluster a descriptive label based on its profile."
        )
        lines.append("")

    if context:
        lines.append(f"Context: {context}")

    return "\n".join(lines)


def _interpret_pca(result: Any, context: str) -> str:
    """Interpret a PCAResult from pyanalytica.model.reduce."""
    lines: list[str] = []

    lines.append("MODEL: Principal Component Analysis (PCA)")
    lines.append("")
    lines.append(
        "PCA is a dimensionality-reduction technique.  It transforms a "
        "set of correlated variables into a smaller set of UNCORRELATED "
        "'principal components' that capture as much of the original "
        "variation as possible."
    )
    lines.append("")

    explained = getattr(result, "explained_variance", [])
    cumulative = getattr(result, "cumulative_variance", [])
    recommended_n = getattr(result, "recommended_n", None)

    if explained:
        lines.append("VARIANCE EXPLAINED PER COMPONENT:")
        for i, v in enumerate(explained):
            bar = "#" * int(v * 50)
            cum = cumulative[i] if i < len(cumulative) else "?"
            lines.append(
                f"  PC{i + 1}: {v:.1%}  (cumulative: {cum:.1%})  {bar}"
            )
        lines.append("")

    if recommended_n is not None:
        lines.append(
            f"RECOMMENDATION: Retain {recommended_n} component(s) to "
            f"capture at least 80% of the total variance."
        )
        if recommended_n == 1:
            lines.append(
                "With only 1 component needed, most variables are telling "
                "a very similar story -- they are highly correlated."
            )
        lines.append("")

    # Loadings
    loadings = getattr(result, "loadings", None)
    if loadings is not None and hasattr(loadings, "to_string") and len(loadings) > 0:
        lines.append("COMPONENT LOADINGS (how each variable contributes):")
        lines.append(loadings.to_string())
        lines.append("")
        lines.append(
            "Loadings close to +1 or -1 mean the variable is strongly "
            "associated with that component.  Variables with similar "
            "loading patterns 'move together' in the data."
        )
        lines.append("")

    if context:
        lines.append(f"Context: {context}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback interpreter
# ---------------------------------------------------------------------------

def _interpret_generic(result: Any, context: str) -> str:
    """Best-effort interpretation for an unknown result type."""
    lines: list[str] = []
    type_name = type(result).__name__
    lines.append(f"RESULT: {type_name}")
    lines.append("")

    # Try common attribute names
    for attr in ("interpretation", "summary", "description"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and val.strip():
            lines.append(val)
            lines.append("")
            break

    for attr in ("p_value", "pvalue"):
        p = getattr(result, attr, None)
        if p is not None:
            lines.append(f"p-value = {_fmt_p(p)}")
            lines.append(_significance_statement(p))
            lines.append("")
            break

    for attr in ("r_squared", "r2", "score"):
        val = getattr(result, attr, None)
        if val is not None:
            lines.append(f"{attr} = {val}")
            lines.append("")
            break

    if not lines or (len(lines) == 2 and lines[1] == ""):
        lines.append(
            "This result type is not yet recognized by the rule-based "
            "interpreter.  Try using the LLM-enhanced mode by setting "
            "the ANTHROPIC_API_KEY environment variable, or inspect the "
            "result object directly."
        )

    if context:
        lines.append(f"Context: {context}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher table
# ---------------------------------------------------------------------------

_INTERPRETERS: dict[str, Any] = {
    "MeansTestResult": _interpret_means_test,
    "CorrelationResult": _interpret_correlation,
    "ProportionsResult": _interpret_proportions,
    "RegressionResult": _interpret_regression,
    "ClassificationResult": _interpret_classification,
    "ClusterResult": _interpret_cluster,
    "PCAResult": _interpret_pca,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpret_result(result: Any, context: str = "") -> str:
    """Generate a plain-English interpretation of an analytics result.

    Parameters
    ----------
    result : Any
        A result object produced by one of the pyanalytica analyze or
        model functions (e.g., ``MeansTestResult``, ``CorrelationResult``,
        ``RegressionResult``, etc.).
    context : str, optional
        Additional context about the data or analysis (e.g., "this is
        survey data from first-year MBA students").  Used to enrich both
        the rule-based and LLM-enhanced outputs.

    Returns
    -------
    str
        A multi-line plain-English explanation suitable for business
        school students.

    Notes
    -----
    The function operates in two modes:

    1. **Rule-based** (always available): uses ``type(result).__name__``
       to dispatch to a specialized interpreter.  No external packages or
       API keys required.
    2. **LLM-enhanced** (optional): if the ``ANTHROPIC_API_KEY`` environment
       variable is set and the ``anthropic`` package is installed, the
       rule-based interpretation is sent to Claude for a richer, more
       nuanced response.  The LLM output is appended after the rule-based
       output so students always have a reliable baseline.
    """
    type_name = type(result).__name__
    interpreter = _INTERPRETERS.get(type_name, _interpret_generic)

    # Rule-based interpretation (always runs)
    rule_based = interpreter(result, context)

    # Attempt LLM enhancement
    llm_prompt = (
        "You are a patient statistics tutor for business school students. "
        "A student just ran a statistical analysis and got this result. "
        "Provide a clear, educational explanation that helps them understand "
        "what the numbers mean and what they should look for.\n\n"
        f"Rule-based interpretation:\n{rule_based}\n\n"
    )
    if context:
        llm_prompt += f"Additional context: {context}\n\n"
    llm_prompt += (
        "Enhance this interpretation with additional insights, analogies, "
        "or practical implications.  Keep the tone friendly and educational. "
        "Do NOT contradict the rule-based interpretation -- add to it."
    )

    llm_response = _try_llm(llm_prompt)
    if llm_response:
        return (
            rule_based
            + "\n\n"
            + "=" * 60
            + "\nAI-ENHANCED INTERPRETATION (powered by Claude):\n"
            + "=" * 60
            + "\n"
            + llm_response
        )

    return rule_based
