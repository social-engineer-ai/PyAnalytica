"""AI-powered next-step suggestion engine for the analytics workbench.

Analyzes the user's operation history and (optionally) the current
DataFrame to recommend what to do next.  Operates in two modes:

1. **Rule-based** (always available): pattern-matches on the history of
   operations to suggest a logical next step in the analytics workflow.
2. **LLM-enhanced** (optional): when an Anthropic API key is set, sends
   the context to Claude for a more tailored recommendation.

The suggestions follow a natural analytics workflow:
  load -> profile -> clean -> explore -> analyze -> model -> evaluate

Usage:
    from pyanalytica.ai.suggest import suggest_next_step
    suggestion = suggest_next_step(workbench.history, current_df=df)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pyanalytica.ai._llm import try_llm as _try_llm


# ---------------------------------------------------------------------------
# DataFrame analysis helpers
# ---------------------------------------------------------------------------

def _describe_df(df: pd.DataFrame) -> str:
    """Build a compact summary of a DataFrame for suggestion context."""
    lines: list[str] = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Column types
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if numeric_cols:
        lines.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:8])}")
        if len(numeric_cols) > 8:
            lines.append(f"  ... and {len(numeric_cols) - 8} more")
    if categorical_cols:
        lines.append(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:8])}")
        if len(categorical_cols) > 8:
            lines.append(f"  ... and {len(categorical_cols) - 8} more")

    # Missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        pct_missing = (cols_with_missing / len(df) * 100).round(1)
        lines.append(f"Columns with missing values: {len(cols_with_missing)}")
        for col_name in cols_with_missing.index[:5]:
            lines.append(
                f"  - {col_name}: {cols_with_missing[col_name]} missing "
                f"({pct_missing[col_name]:.1f}%)"
            )
    else:
        lines.append("No missing values detected.")

    return "\n".join(lines)


def _get_action_set(history: list) -> set[str]:
    """Extract the set of unique action types from history."""
    actions: set[str] = set()
    for op in history:
        action = getattr(op, "action", None)
        if action and isinstance(action, str):
            actions.add(action.lower())
    return actions


def _get_last_action(history: list) -> str:
    """Get the most recent action from history."""
    for op in reversed(history):
        action = getattr(op, "action", None)
        if action and isinstance(action, str):
            return action.lower()
    return ""


# ---------------------------------------------------------------------------
# Rule-based suggestion engine
# ---------------------------------------------------------------------------

def _rule_based_suggestion(
    history: list, df: pd.DataFrame | None
) -> str:
    """Generate a next-step suggestion based on the operation history.

    The suggestion follows the natural analytics workflow:
      load -> profile -> clean/transform -> explore -> analyze -> model -> evaluate
    """
    if not history:
        return (
            "NEXT STEP: Load a dataset\n\n"
            "Start by loading a dataset into the workbench.  You can:\n"
            "  - Upload a CSV or Excel file\n"
            "  - Use one of the built-in sample datasets\n"
            "  - Connect to a data source\n\n"
            "Once your data is loaded, we will guide you through profiling, "
            "cleaning, exploration, analysis, and modeling."
        )

    actions = _get_action_set(history)
    last_action = _get_last_action(history)

    # Build a comprehensive suggestion based on workflow stage
    lines: list[str] = []

    # Stage 1: Just loaded, haven't profiled
    if actions <= {"load"} or last_action == "load":
        lines.append("NEXT STEP: Profile your data")
        lines.append("")
        lines.append(
            "Great -- your data is loaded!  Before diving into analysis, "
            "take a moment to understand its structure:"
        )
        lines.append("  1. Check the Data Profile to see column types, "
                      "distributions, and summary statistics")
        lines.append("  2. Look for missing values, outliers, and "
                      "unexpected patterns")
        lines.append("  3. Identify which columns are numeric vs. categorical")
        if df is not None:
            lines.append("")
            lines.append("YOUR DATA:")
            lines.append(_describe_df(df))
        return "\n".join(lines)

    # Stage 2: Profiled, haven't cleaned
    if actions <= {"load", "profile"} or last_action == "profile":
        lines.append("NEXT STEP: Clean and prepare your data")
        lines.append("")
        lines.append(
            "Now that you understand your data's structure, address any "
            "data quality issues:"
        )
        lines.append("  1. Handle missing values (drop rows, impute, "
                      "or flag them)")
        lines.append("  2. Fix data types if needed (e.g., convert strings "
                      "to numbers)")
        lines.append("  3. Remove duplicates if present")
        lines.append("  4. Consider creating new variables or recoding "
                      "existing ones")
        if df is not None:
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing > 0:
                lines.append("")
                lines.append(
                    f"NOTE: Your dataset has {total_missing} missing values "
                    f"across {(missing > 0).sum()} columns. Address these "
                    f"before analysis."
                )
            else:
                lines.append("")
                lines.append(
                    "Your data looks clean -- no missing values detected. "
                    "You may proceed to exploration."
                )
        return "\n".join(lines)

    # Stage 3: Cleaned/transformed, haven't explored
    if (
        actions & {"transform", "clean", "filter", "merge", "recode"}
        and not actions & {"visualize", "explore", "analyze", "model"}
    ) or last_action in ("transform", "clean", "filter", "merge", "recode"):
        lines.append("NEXT STEP: Explore your data")
        lines.append("")
        lines.append(
            "Your data is cleaned and ready for exploration.  Try these:"
        )
        lines.append("  1. Compute summary statistics (mean, median, "
                      "standard deviation)")
        lines.append("  2. Create visualizations: histograms for "
                      "distributions, scatter plots for relationships, "
                      "box plots to compare groups")
        lines.append("  3. Look for interesting patterns or relationships "
                      "between variables")
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if len(numeric_cols) >= 2:
                lines.append("")
                lines.append(
                    f"TIP: You have {len(numeric_cols)} numeric columns -- "
                    f"try a scatter plot or correlation matrix to see how "
                    f"they relate."
                )
            if cat_cols and numeric_cols:
                lines.append(
                    f"TIP: You have both categorical ({cat_cols[0]}) and "
                    f"numeric ({numeric_cols[0]}) columns -- try a box plot "
                    f"to compare groups."
                )
        return "\n".join(lines)

    # Stage 4: Explored/visualized, haven't analyzed
    if (
        actions & {"visualize", "explore"}
        and not actions & {"analyze", "model"}
    ) or last_action in ("visualize", "explore"):
        lines.append("NEXT STEP: Run a statistical test")
        lines.append("")
        lines.append(
            "You have explored your data and seen some patterns.  Now let's "
            "test whether those patterns are statistically significant:"
        )
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if cat_cols and numeric_cols:
                lines.append(
                    f"  - COMPARE MEANS: Use a t-test or ANOVA to compare "
                    f"'{numeric_cols[0]}' across groups in '{cat_cols[0]}'"
                )
            if len(numeric_cols) >= 2:
                lines.append(
                    f"  - CORRELATION: Test the relationship between "
                    f"'{numeric_cols[0]}' and '{numeric_cols[1]}'"
                )
            if len(cat_cols) >= 2:
                lines.append(
                    f"  - CHI-SQUARE: Test the association between "
                    f"'{cat_cols[0]}' and '{cat_cols[1]}'"
                )
        else:
            lines.append("  - Compare means (t-test, ANOVA)")
            lines.append("  - Test correlations")
            lines.append("  - Chi-square test for categorical variables")
        return "\n".join(lines)

    # Stage 5: Analyzed, haven't modeled
    if (
        actions & {"analyze"}
        and not actions & {"model"}
    ) or last_action == "analyze":
        lines.append("NEXT STEP: Build a predictive model")
        lines.append("")
        lines.append(
            "Your statistical tests have revealed significant relationships. "
            "Now try building a predictive model:"
        )
        lines.append("  - LINEAR REGRESSION: Predict a numeric outcome "
                      "from multiple predictors")
        lines.append("  - LOGISTIC REGRESSION: Predict a categorical "
                      "outcome (yes/no, class A/B/C)")
        lines.append("  - DECISION TREE: Build an interpretable "
                      "rule-based classifier")
        lines.append("  - CLUSTERING: Discover natural groups in your data "
                      "(unsupervised)")
        lines.append("  - PCA: Reduce the number of variables while "
                      "retaining information")
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) >= 3:
                lines.append("")
                lines.append(
                    f"TIP: With {len(numeric_cols)} numeric columns, "
                    f"regression or PCA would be good starting points."
                )
        return "\n".join(lines)

    # Stage 6: Modeled, suggest evaluation or alternative
    if actions & {"model"} or last_action == "model":
        lines.append("NEXT STEP: Evaluate and iterate")
        lines.append("")
        lines.append(
            "You have built a model!  Here are productive next steps:"
        )
        lines.append("  1. EVALUATE: Check your model's accuracy, "
                      "residuals, and assumptions")
        lines.append("  2. COMPARE: Try a different model type and compare "
                      "performance")
        lines.append("  3. INTERPRET: Use the AI interpreter to get a "
                      "plain-English explanation of your results")
        lines.append("  4. REPORT: Generate a summary report of your "
                      "full analysis")
        lines.append("  5. REFINE: Go back and try different variables, "
                      "transformations, or parameters")
        return "\n".join(lines)

    # Stage 7: Evaluated
    if actions & {"evaluate"} or last_action == "evaluate":
        lines.append("NEXT STEP: Refine or report")
        lines.append("")
        lines.append(
            "Your model has been evaluated.  Consider:"
        )
        lines.append("  1. Try different features or transformations to "
                      "improve performance")
        lines.append("  2. Compare multiple models side by side")
        lines.append("  3. Generate a final report summarizing your "
                      "findings")
        lines.append("  4. Export your results and share with stakeholders")
        return "\n".join(lines)

    # Default: general guidance
    lines.append("NEXT STEP: Continue your analysis")
    lines.append("")
    lines.append(
        "You have completed several steps in the analytics workflow. "
        "Here are some options:"
    )
    lines.append("  - Explore a different aspect of the data")
    lines.append("  - Try a new statistical test or model")
    lines.append("  - Generate a report of your findings")
    lines.append("  - Load additional data to enrich your analysis")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_next_step(
    history: list,
    current_df: pd.DataFrame | None = None,
) -> str:
    """Suggest the next analytical step based on the user's history.

    Parameters
    ----------
    history : list
        A list of ``Operation`` objects (from ``pyanalytica.core.state``)
        representing what the user has done so far.  Each object has an
        ``.action`` attribute (e.g., ``"load"``, ``"transform"``,
        ``"visualize"``, ``"analyze"``, ``"model"``).
    current_df : pd.DataFrame or None, optional
        The currently active DataFrame.  When provided, the suggestion
        engine uses column types, missing-value counts, and shape to
        give more specific recommendations.

    Returns
    -------
    str
        A multi-line plain-English suggestion describing what to do next,
        with concrete examples where possible.

    Notes
    -----
    Operates in two modes:

    1. **Rule-based** (always available): follows a logical analytics
       workflow progression based on what actions have been performed.
    2. **LLM-enhanced** (optional): when ``ANTHROPIC_API_KEY`` is set,
       sends the history and data summary to Claude for a richer,
       more tailored recommendation.
    """
    # Rule-based suggestion (always runs)
    rule_based = _rule_based_suggestion(history, current_df)

    # Build LLM prompt
    history_summary = []
    for op in history[-20:]:  # Last 20 operations at most
        action = getattr(op, "action", "?")
        desc = getattr(op, "description", "")
        history_summary.append(f"  - [{action}] {desc}")

    llm_prompt = (
        "You are an analytics tutor for business school students. "
        "Based on their analysis history, suggest ONE clear next step.\n\n"
        f"Operation history (most recent last):\n"
        + "\n".join(history_summary)
        + "\n\n"
    )
    if current_df is not None:
        llm_prompt += f"Current dataset:\n{_describe_df(current_df)}\n\n"
    llm_prompt += (
        f"Rule-based suggestion:\n{rule_based}\n\n"
        "Enhance this suggestion with specific, actionable advice. "
        "Be encouraging and educational. Keep it concise (3-5 sentences)."
    )

    llm_response = _try_llm(llm_prompt)
    if llm_response:
        return (
            rule_based
            + "\n\n"
            + "-" * 40
            + "\nAI SUGGESTION (powered by Claude):\n"
            + "-" * 40
            + "\n"
            + llm_response
        )

    return rule_based
