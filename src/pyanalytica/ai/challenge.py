"""Socratic questioning engine for challenging student interpretations.

This module NEVER gives the answer directly.  Instead, it generates
probing questions that guide students to discover errors or deepen
their understanding on their own.

Operates in two modes:

1. **Rule-based** (always available): pattern-matches on the student's
   text and the result object to identify common misconceptions and
   generate targeted Socratic questions.
2. **LLM-enhanced** (optional): when an Anthropic API key is set,
   sends the student's interpretation to Claude for more nuanced,
   context-aware questioning.

Usage:
    from pyanalytica.ai.challenge import challenge_interpretation
    questions = challenge_interpretation(
        student_text="The result is significant so X causes Y",
        result=my_correlation_result,
        context="marketing spend vs. sales"
    )
"""

from __future__ import annotations

import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# LLM helper (optional)
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
            max_tokens=768,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

def _text_lower(text: str) -> str:
    """Return lowercased text for pattern matching."""
    return text.lower().strip()


def _mentions(text: str, *keywords: str) -> bool:
    """Check if the text mentions any of the given keywords."""
    text_lower = _text_lower(text)
    return any(kw.lower() in text_lower for kw in keywords)


def _get_p_value(result: Any) -> float | None:
    """Extract p-value from a result object."""
    for attr in ("p_value", "pvalue", "f_pvalue"):
        val = getattr(result, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None


def _get_r_value(result: Any) -> float | None:
    """Extract correlation coefficient from a result object."""
    val = getattr(result, "r", None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def _get_effect_size(result: Any) -> float | None:
    """Extract effect size from a result object."""
    val = getattr(result, "effect_size", None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def _get_r_squared(result: Any) -> float | None:
    """Extract R-squared from a result object."""
    val = getattr(result, "r_squared", None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def _get_train_accuracy(result: Any) -> float | None:
    """Extract training accuracy from a result object."""
    val = getattr(result, "train_accuracy", None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def _get_test_accuracy(result: Any) -> float | None:
    """Extract test accuracy from a result object."""
    val = getattr(result, "test_accuracy", None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


# ---------------------------------------------------------------------------
# Rule-based challenge generators
# ---------------------------------------------------------------------------

def _challenge_significance_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about statistical significance."""
    p_value = _get_p_value(result)
    if p_value is None:
        return

    text_lower = _text_lower(text)

    # Student says significant but p > 0.05
    if (
        _mentions(text, "significant", "reject")
        and p_value > 0.05
        and not _mentions(text, "not significant", "insignificant",
                          "fail to reject", "failed to reject",
                          "do not reject", "cannot reject")
    ):
        questions.append(
            "You mentioned that the result is significant. "
            "Take another look at the p-value. What is the conventional "
            "threshold for statistical significance, and how does your "
            "p-value compare to it?"
        )

    # Student says not significant but p < 0.05
    if (
        _mentions(text, "not significant", "insignificant",
                  "fail to reject", "no effect", "no difference",
                  "no relationship")
        and p_value < 0.05
    ):
        questions.append(
            "You concluded that the result is not significant. "
            "Look at the p-value again carefully -- is it above or below "
            "0.05? What does that tell you about the null hypothesis?"
        )

    # Student confuses practical and statistical significance
    if _mentions(text, "significant") and not _mentions(text, "practical", "effect size"):
        es = _get_effect_size(result)
        if es is not None and es < 0.2:
            questions.append(
                "You noted statistical significance, which is great. "
                "But is the effect actually meaningful in practical terms? "
                "What does the effect size tell you about the real-world "
                "importance of this finding?"
            )

    # Student doesn't mention p-value at all
    if not _mentions(text, "p-value", "p value", "p=", "p =", "p <", "significance"):
        questions.append(
            "Your interpretation does not mention the p-value. "
            "What role does the p-value play in deciding whether to "
            "accept or reject the null hypothesis?"
        )


def _challenge_correlation_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about correlation results."""
    type_name = type(result).__name__
    if type_name != "CorrelationResult":
        return

    r = _get_r_value(result)
    text_lower = _text_lower(text)

    # Causation from correlation
    if _mentions(text, "cause", "causes", "caused", "because of",
                 "leads to", "results in", "due to", "effect of"):
        questions.append(
            "You used causal language (e.g., 'causes,' 'leads to'). "
            "Can we infer causation from a correlation alone? "
            "What kind of study design would we need to establish "
            "a causal relationship?"
        )

    # Ignoring direction of correlation
    if r is not None and not _mentions(text, "positive", "negative",
                                        "direct", "inverse"):
        questions.append(
            "You discussed the correlation but did not mention its "
            "direction. Is the relationship positive or negative? "
            "What does that direction mean in practical terms for "
            "these variables?"
        )

    # Misidentifying strength
    if r is not None:
        abs_r = abs(r)
        if abs_r < 0.3 and _mentions(text, "strong"):
            questions.append(
                "You described this correlation as 'strong.' "
                "With r = {:.3f}, where does this fall on the typical "
                "scale of correlation strength? What r values are "
                "generally considered 'strong'?".format(r)
            )
        if abs_r > 0.7 and _mentions(text, "weak"):
            questions.append(
                "You described this correlation as 'weak.' "
                "With r = {:.3f}, does 'weak' accurately characterize "
                "this magnitude? What is the conventional scale?".format(r)
            )

    # Not mentioning sample size
    n = getattr(result, "n", None)
    if n is not None and not _mentions(text, "sample size", "n =",
                                        "observations", "data points"):
        questions.append(
            "How might the sample size (n = {}) affect your confidence "
            "in this correlation? Would you feel differently about this "
            "result if n were much larger or smaller?".format(n)
        )


def _challenge_regression_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about regression results."""
    type_name = type(result).__name__
    if type_name != "RegressionResult":
        return

    r_sq = _get_r_squared(result)

    # Overinterpreting R-squared
    if r_sq is not None and r_sq < 0.3 and _mentions(text, "good fit",
                                                       "explains well",
                                                       "strong model",
                                                       "accurate"):
        questions.append(
            "You described the model as a good fit. "
            "The R-squared is {:.3f}, meaning the model explains {:.1f}% "
            "of the variance. Is that enough to call it a 'good' fit? "
            "What might account for the unexplained {:.1f}%?".format(
                r_sq, r_sq * 100, (1 - r_sq) * 100
            )
        )

    # Not discussing assumptions
    if not _mentions(text, "assumption", "residual", "normality",
                     "heteroscedasticity", "linearity", "multicollinearity",
                     "VIF"):
        questions.append(
            "What assumptions does linear regression make about the data? "
            "How would you check whether those assumptions are met here?"
        )

    # Causation from regression
    if _mentions(text, "cause", "causes", "impact", "effect of"):
        questions.append(
            "You used language suggesting a causal relationship. "
            "Does a regression coefficient represent a causal effect? "
            "Under what conditions could we interpret it causally?"
        )

    # Ignoring multicollinearity
    vif_df = getattr(result, "vif", None)
    if vif_df is not None and hasattr(vif_df, "iterrows"):
        high_vif = any(
            row.get("VIF", 0) > 5
            for _, row in vif_df.iterrows()
        )
        if high_vif and not _mentions(text, "collinear", "VIF",
                                       "multicollinear"):
            questions.append(
                "The VIF table shows some high values. What is "
                "multicollinearity and how might it affect the "
                "interpretation of individual coefficients?"
            )


def _challenge_classification_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about classification results."""
    type_name = type(result).__name__
    if type_name != "ClassificationResult":
        return

    train_acc = _get_train_accuracy(result)
    test_acc = _get_test_accuracy(result)

    # Not mentioning overfitting when gap is large
    if train_acc is not None and test_acc is not None:
        gap = train_acc - test_acc
        if gap > 0.10 and not _mentions(text, "overfit", "over-fit",
                                         "memoriz", "generaliz"):
            questions.append(
                "The training accuracy is {:.1%} but the test accuracy is "
                "{:.1%}. What might explain this gap? What concept in "
                "machine learning describes this situation?".format(
                    train_acc, test_acc
                )
            )

    # Only focusing on accuracy
    if _mentions(text, "accuracy", "accurate") and not _mentions(
        text, "precision", "recall", "f1", "false positive",
        "false negative", "confusion matrix", "class imbalance",
        "balanced"
    ):
        questions.append(
            "You focused on accuracy. Is accuracy always the best metric? "
            "What happens if the classes are imbalanced (e.g., 95% of "
            "observations are in one class)? What other metrics might "
            "be informative?"
        )

    # Feature importance
    fi = getattr(result, "feature_importance", None)
    if fi is not None and not _mentions(text, "feature", "important",
                                         "variable", "predictor"):
        questions.append(
            "The model identified which features are most important. "
            "Which variables contribute most to the model's predictions? "
            "Does that make intuitive sense?"
        )


def _challenge_cluster_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about clustering results."""
    type_name = type(result).__name__
    if type_name != "ClusterResult":
        return

    # Not questioning k
    n_clusters = getattr(result, "n_clusters", None)
    if n_clusters is not None and not _mentions(text, "number of cluster",
                                                 "how many", "elbow",
                                                 "silhouette", "choice of k",
                                                 "optimal k"):
        questions.append(
            "You accepted {} clusters. How do you know that is the right "
            "number? What methods can help determine the optimal number "
            "of clusters?".format(n_clusters)
        )

    # Not interpreting clusters
    if not _mentions(text, "profile", "characteriz", "interpret",
                     "describe", "label", "meaning"):
        questions.append(
            "What makes these clusters different from each other? "
            "Can you describe each cluster in plain language -- "
            "what 'type' of observation does each cluster represent?"
        )

    # Silhouette score
    sil = getattr(result, "silhouette_scores", [])
    if sil and not _mentions(text, "silhouette", "quality", "well-separated"):
        questions.append(
            "How well-separated are the clusters? "
            "What does the silhouette score tell you about cluster quality, "
            "and how should it influence your confidence in these groupings?"
        )


def _challenge_pca_claims(
    text: str, result: Any, questions: list[str]
) -> None:
    """Challenge claims about PCA results."""
    type_name = type(result).__name__
    if type_name != "PCAResult":
        return

    # Not discussing variance explained
    if not _mentions(text, "variance", "explain", "information"):
        questions.append(
            "How much of the total variance do the retained components "
            "capture? Is that enough to feel confident we have not lost "
            "important information?"
        )

    # Not discussing loadings
    if not _mentions(text, "loading", "contribut", "weight"):
        questions.append(
            "What do the component loadings tell us? "
            "Which original variables contribute most to each principal "
            "component, and what might that mean conceptually?"
        )

    # Recommended n
    recommended_n = getattr(result, "recommended_n", None)
    explained = getattr(result, "explained_variance", [])
    if recommended_n is not None and len(explained) > recommended_n:
        if not _mentions(text, "component", "dimension", "retain", "keep"):
            questions.append(
                "How many components should we retain? "
                "What criteria did you use (e.g., eigenvalue > 1, scree "
                "plot elbow, cumulative variance threshold)?"
            )


def _add_generic_challenges(
    text: str, result: Any, questions: list[str]
) -> None:
    """Add generic Socratic questions that apply to most analyses."""
    type_name = type(result).__name__

    # Assumptions
    if (
        type_name in ("MeansTestResult", "CorrelationResult",
                       "ProportionsResult", "RegressionResult")
        and not _mentions(text, "assumption", "condition", "requirement",
                          "prerequisite")
        and len(questions) < 2  # Don't overwhelm with too many questions
    ):
        questions.append(
            "What assumptions does this test make about the data? "
            "Have those assumptions been checked?"
        )

    # Sample size considerations
    if (
        not _mentions(text, "sample size", "n =", "power", "observations")
        and len(questions) < 3
    ):
        questions.append(
            "How might the sample size affect your conclusion? "
            "Would a larger or smaller sample change your interpretation?"
        )

    # Alternative significance levels
    p_value = _get_p_value(result)
    if (
        p_value is not None
        and 0.01 < p_value < 0.10
        and not _mentions(text, "alpha", "0.01", "0.10",
                          "significance level", "threshold")
        and len(questions) < 4
    ):
        questions.append(
            "What would change if we used a different significance level "
            "(e.g., 0.01 instead of 0.05)? Why might different fields "
            "or contexts use different thresholds?"
        )

    # Always ensure at least one question
    if not questions:
        questions.append(
            "That is a good start.  Can you elaborate on your reasoning?  "
            "What specific numbers from the output support your conclusion?"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def challenge_interpretation(
    student_text: str,
    result: Any,
    context: str = "",
) -> str:
    """Generate Socratic questions to challenge a student's interpretation.

    This function NEVER gives the answer directly.  Instead, it asks
    probing questions that guide students to discover errors or deepen
    their understanding on their own.

    Parameters
    ----------
    student_text : str
        The student's written interpretation of the analysis result.
    result : Any
        The result object from a pyanalytica analysis (e.g.,
        ``MeansTestResult``, ``CorrelationResult``, etc.).
    context : str, optional
        Additional context about the analysis (e.g., the research
        question or dataset description).

    Returns
    -------
    str
        A multi-line string containing numbered Socratic questions.
        These questions are designed to be educational, never punitive.

    Notes
    -----
    Operates in two modes:

    1. **Rule-based** (always available): pattern-matches on the student
       text and result attributes to detect common misconceptions.
    2. **LLM-enhanced** (optional): when ``ANTHROPIC_API_KEY`` is set,
       sends the student's text and result to Claude for more nuanced,
       context-aware Socratic questioning.
    """
    if not student_text or not student_text.strip():
        return (
            "It looks like you have not written an interpretation yet.\n\n"
            "Try writing 2-3 sentences about what the results mean. "
            "Consider:\n"
            "  1. What does the test statistic or main metric tell you?\n"
            "  2. Is the result statistically significant?\n"
            "  3. What is the practical implication?"
        )

    questions: list[str] = []

    # Run all rule-based challengers
    _challenge_significance_claims(student_text, result, questions)
    _challenge_correlation_claims(student_text, result, questions)
    _challenge_regression_claims(student_text, result, questions)
    _challenge_classification_claims(student_text, result, questions)
    _challenge_cluster_claims(student_text, result, questions)
    _challenge_pca_claims(student_text, result, questions)
    _add_generic_challenges(student_text, result, questions)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_questions: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    # Cap at 5 questions to avoid overwhelming the student
    unique_questions = unique_questions[:5]

    # Format output
    lines: list[str] = []
    lines.append("QUESTIONS TO DEEPEN YOUR UNDERSTANDING:")
    lines.append("")
    for i, q in enumerate(unique_questions, 1):
        lines.append(f"  {i}. {q}")
        lines.append("")

    lines.append(
        "Take a moment to think about each question, then revise your "
        "interpretation if needed.  There are no wrong answers in the "
        "learning process -- the goal is to think critically about "
        "what the data tells us."
    )

    rule_based = "\n".join(lines)

    # Attempt LLM enhancement
    llm_prompt = (
        "You are a Socratic statistics tutor. A student wrote this "
        "interpretation of their analysis result. Your job is to ask "
        "probing questions that help them think deeper -- NEVER give "
        "the answer directly.\n\n"
        f"Student's interpretation:\n\"{student_text}\"\n\n"
        f"Result type: {type(result).__name__}\n"
    )

    # Add key result attributes for context
    p_value = _get_p_value(result)
    if p_value is not None:
        llm_prompt += f"p-value: {p_value}\n"
    r = _get_r_value(result)
    if r is not None:
        llm_prompt += f"Correlation r: {r}\n"
    r_sq = _get_r_squared(result)
    if r_sq is not None:
        llm_prompt += f"R-squared: {r_sq}\n"

    if context:
        llm_prompt += f"\nContext: {context}\n"

    llm_prompt += (
        f"\nRule-based questions already generated:\n{rule_based}\n\n"
        "Add 1-2 additional thought-provoking Socratic questions that "
        "the rule-based system missed. Focus on deeper conceptual "
        "understanding. Be encouraging, not critical."
    )

    llm_response = _try_llm(llm_prompt)
    if llm_response:
        return (
            rule_based
            + "\n\n"
            + "-" * 40
            + "\nADDITIONAL QUESTIONS (powered by Claude):\n"
            + "-" * 40
            + "\n"
            + llm_response
        )

    return rule_based
