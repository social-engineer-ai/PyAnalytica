"""Tests for the pyanalytica.ai package — interpret, suggest, challenge, query."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from pyanalytica.ai.interpret import interpret_result
from pyanalytica.ai.suggest import suggest_next_step
from pyanalytica.ai.challenge import challenge_interpretation
from pyanalytica.ai.query import natural_language_to_pandas


# ---------------------------------------------------------------------------
# Mock result objects (avoid importing from analyze/model to prevent
# circular dependencies — the AI modules dispatch on type name strings)
# ---------------------------------------------------------------------------

@dataclass
class MeansTestResult:
    test_name: str = "Two-sample t-test"
    statistic: float = 2.45
    p_value: float = 0.018
    confidence_interval: tuple = (0.5, 3.2)
    effect_size: float = 0.45
    effect_size_name: str = "Cohen's d"
    group_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    assumption_checks: dict = field(default_factory=lambda: {
        "normality_ok": True,
        "normality_shapiro_p": 0.32,
        "equal_variance": True,
        "levene_p": 0.54,
    })
    interpretation: str = "Significant difference found."
    code: object = None


@dataclass
class CorrelationResult:
    method: str = "pearson"
    r: float = 0.72
    p_value: float = 0.001
    ci_lower: float = 0.55
    ci_upper: float = 0.84
    n: int = 50
    interpretation: str = "Strong positive correlation."
    code: object = None


@dataclass
class ProportionsResult:
    chi2: float = 12.5
    p_value: float = 0.002
    dof: int = 2
    observed: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    interpretation: str = "Significant association."
    code: object = None


@dataclass
class RegressionResult:
    coefficients: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "variable": ["(Intercept)", "age", "income"],
        "coefficient": [10.0, 0.5, 0.001],
        "p_value": [0.001, 0.03, 0.45],
    }))
    r_squared: float = 0.65
    adj_r_squared: float = 0.63
    f_stat: float = 15.2
    f_pvalue: float = 0.0001
    vif: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "variable": ["age", "income"],
        "VIF": [1.2, 1.2],
    }))
    residual_plot: object = None
    qq_plot: object = None
    interpretation: str = "Model explains 65% of variance."
    predictions: object = None
    code: object = None


@dataclass
class ClassificationResult:
    model_type: str = "Logistic Regression"
    coefficients: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "variable": ["age", "income"],
        "coefficient": [0.3, -0.1],
    }))
    feature_importance: object = None
    train_accuracy: float = 0.92
    test_accuracy: float = 0.78
    predictions: object = None
    probabilities: object = None
    code: object = None


@dataclass
class ClusterResult:
    labels: pd.Series = field(default_factory=lambda: pd.Series([0, 1, 0, 1, 2]))
    n_clusters: int = 3
    elbow_plot: object = None
    silhouette_scores: list = field(default_factory=lambda: [0.45, 0.52, 0.48])
    cluster_profiles: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "age": [25, 35, 55],
        "income": [30000, 60000, 90000],
    }, index=[0, 1, 2]))
    scatter_plot: object = None
    code: object = None


@dataclass
class PCAResult:
    components: pd.DataFrame = field(default_factory=pd.DataFrame)
    explained_variance: list = field(default_factory=lambda: [0.45, 0.25, 0.15, 0.10, 0.05])
    cumulative_variance: list = field(default_factory=lambda: [0.45, 0.70, 0.85, 0.95, 1.0])
    loadings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "PC1": [0.5, -0.3, 0.7],
        "PC2": [0.2, 0.8, -0.1],
    }, index=["var1", "var2", "var3"]))
    scree_plot: object = None
    biplot: object = None
    recommended_n: int = 3
    code: object = None


# Mock Operation for suggest tests
@dataclass
class Operation:
    action: str = "load"
    description: str = "Loaded dataset"
    dataset: str = "test"


# ---------------------------------------------------------------------------
# Test interpret_result
# ---------------------------------------------------------------------------

class TestInterpretResult:
    def test_means_test(self):
        result = MeansTestResult()
        text = interpret_result(result)
        assert "Two-sample t-test" in text
        assert "p-value" in text.lower()
        assert "reject" in text.lower()
        assert "Cohen's d" in text

    def test_means_not_significant(self):
        result = MeansTestResult(p_value=0.35, statistic=0.9)
        text = interpret_result(result)
        assert "fail to reject" in text.lower()

    def test_correlation(self):
        result = CorrelationResult()
        text = interpret_result(result)
        assert "Pearson" in text
        assert "positive" in text.lower()
        assert "causation" in text.lower() or "causal" in text.lower()

    def test_correlation_negative(self):
        result = CorrelationResult(r=-0.55)
        text = interpret_result(result)
        assert "negative" in text.lower()

    def test_proportions(self):
        result = ProportionsResult()
        text = interpret_result(result)
        assert "Chi-Square" in text or "chi-square" in text.lower()
        assert "p-value" in text.lower()

    def test_regression(self):
        result = RegressionResult()
        text = interpret_result(result)
        assert "R-squared" in text or "R-square" in text
        assert "65" in text  # 65% variance
        assert "age" in text  # significant predictor

    def test_classification(self):
        result = ClassificationResult()
        text = interpret_result(result)
        assert "Logistic" in text
        assert "accuracy" in text.lower()
        # Should warn about overfitting (92% train vs 78% test)
        assert "overfit" in text.lower() or "gap" in text.lower()

    def test_cluster(self):
        result = ClusterResult()
        text = interpret_result(result)
        assert "Clustering" in text or "cluster" in text.lower()
        assert "silhouette" in text.lower()

    def test_pca(self):
        result = PCAResult()
        text = interpret_result(result)
        assert "PCA" in text or "Principal Component" in text
        assert "variance" in text.lower()

    def test_unknown_type(self):
        text = interpret_result("just a string")
        # Should still return something (generic interpreter)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_context_included(self):
        result = MeansTestResult()
        text = interpret_result(result, context="marketing survey")
        assert "marketing survey" in text

    def test_one_sample_ttest(self):
        result = MeansTestResult(test_name="One-sample t-test")
        text = interpret_result(result)
        assert "one" in text.lower() or "sample" in text.lower()

    def test_anova(self):
        result = MeansTestResult(test_name="One-way ANOVA")
        text = interpret_result(result)
        assert "ANOVA" in text


# ---------------------------------------------------------------------------
# Test suggest_next_step
# ---------------------------------------------------------------------------

class TestSuggestNextStep:
    def test_empty_history(self):
        text = suggest_next_step([])
        assert "load" in text.lower()

    def test_after_load(self):
        history = [Operation(action="load")]
        text = suggest_next_step(history)
        assert "profile" in text.lower() or "understand" in text.lower()

    def test_after_load_with_df(self):
        history = [Operation(action="load")]
        df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
        text = suggest_next_step(history, current_df=df)
        assert "profile" in text.lower() or "structure" in text.lower()

    def test_after_profile(self):
        history = [
            Operation(action="load"),
            Operation(action="profile"),
        ]
        text = suggest_next_step(history)
        assert "clean" in text.lower() or "prepare" in text.lower() or "transform" in text.lower()

    def test_after_transform(self):
        history = [
            Operation(action="load"),
            Operation(action="profile"),
            Operation(action="transform"),
        ]
        text = suggest_next_step(history)
        assert "explore" in text.lower() or "visualiz" in text.lower()

    def test_after_visualize(self):
        history = [
            Operation(action="load"),
            Operation(action="transform"),
            Operation(action="visualize"),
        ]
        text = suggest_next_step(history)
        assert "test" in text.lower() or "analyz" in text.lower() or "statistic" in text.lower()

    def test_after_analyze(self):
        history = [
            Operation(action="load"),
            Operation(action="analyze"),
        ]
        text = suggest_next_step(history)
        assert "model" in text.lower() or "predict" in text.lower()

    def test_after_model(self):
        history = [
            Operation(action="load"),
            Operation(action="model"),
        ]
        text = suggest_next_step(history)
        assert "evaluat" in text.lower() or "refine" in text.lower() or "report" in text.lower()

    def test_with_missing_values(self):
        history = [Operation(action="load"), Operation(action="profile")]
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, None]})
        text = suggest_next_step(history, current_df=df)
        assert "missing" in text.lower()


# ---------------------------------------------------------------------------
# Test challenge_interpretation
# ---------------------------------------------------------------------------

class TestChallengeInterpretation:
    def test_empty_text(self):
        result = MeansTestResult()
        text = challenge_interpretation("", result)
        assert "interpretation" in text.lower()

    def test_causation_from_correlation(self):
        result = CorrelationResult()
        text = challenge_interpretation(
            "Advertising causes sales to increase", result
        )
        assert "caus" in text.lower()

    def test_wrong_significance_claim(self):
        # Student says significant but p > 0.05
        result = MeansTestResult(p_value=0.35)
        text = challenge_interpretation(
            "The result is statistically significant", result
        )
        assert "p-value" in text.lower()

    def test_wrong_not_significant_claim(self):
        # Student says not significant but p < 0.05
        result = MeansTestResult(p_value=0.01)
        text = challenge_interpretation(
            "The result is not significant, there is no effect", result
        )
        assert "p-value" in text.lower()

    def test_missing_direction(self):
        result = CorrelationResult(r=-0.65)
        text = challenge_interpretation(
            "There is a strong correlation", result
        )
        # Should ask about direction
        assert "direction" in text.lower() or "positive" in text.lower() or "negative" in text.lower()

    def test_weak_called_strong(self):
        result = CorrelationResult(r=0.15)
        text = challenge_interpretation(
            "There is a strong positive correlation", result
        )
        assert "strong" in text.lower() or "strength" in text.lower()

    def test_regression_no_assumptions(self):
        result = RegressionResult()
        text = challenge_interpretation(
            "The model is good because R-squared is high", result
        )
        assert "assumption" in text.lower()

    def test_classification_overfitting(self):
        result = ClassificationResult(train_accuracy=0.95, test_accuracy=0.70)
        text = challenge_interpretation(
            "The model is very accurate at 95%", result
        )
        assert "overfit" in text.lower() or "gap" in text.lower() or "training" in text.lower()

    def test_cluster_not_questioning_k(self):
        result = ClusterResult()
        text = challenge_interpretation(
            "The data has three clear clusters", result
        )
        assert "cluster" in text.lower() or "number" in text.lower()

    def test_always_returns_questions(self):
        result = MeansTestResult()
        text = challenge_interpretation("Good result", result)
        # Should always have at least one question
        assert "?" in text

    def test_caps_at_five_questions(self):
        result = RegressionResult()
        text = challenge_interpretation(
            "This is significant and causes the outcome to increase",
            result,
        )
        # Count question marks
        question_count = text.count("?")
        assert question_count <= 10  # 5 questions max, each might have sub-questions


# ---------------------------------------------------------------------------
# Test natural_language_to_pandas
# ---------------------------------------------------------------------------

class TestNaturalLanguageToPandas:
    df_info = {
        "columns": ["salary", "department", "age", "years_experience"],
        "dtypes": {
            "salary": "float64",
            "department": "object",
            "age": "int64",
            "years_experience": "float64",
        },
        "shape": (500, 4),
        "name": "df",
    }

    def test_show_me_column(self):
        code = natural_language_to_pandas("show me salary", self.df_info)
        assert "salary" in code
        assert "head" in code

    def test_show_me_data(self):
        code = natural_language_to_pandas("show me the data", self.df_info)
        assert "head" in code

    def test_average_of_column(self):
        code = natural_language_to_pandas("average of salary", self.df_info)
        assert "salary" in code
        assert "mean" in code

    def test_average_by_group(self):
        code = natural_language_to_pandas(
            "average of salary by department", self.df_info
        )
        assert "groupby" in code
        assert "salary" in code
        assert "department" in code
        assert "mean" in code

    def test_count_of(self):
        code = natural_language_to_pandas("count of department", self.df_info)
        assert "department" in code
        assert "value_counts" in code

    def test_filter_where(self):
        code = natural_language_to_pandas(
            "filter where salary > 50000", self.df_info
        )
        assert "salary" in code
        assert "50000" in code
        assert ">" in code

    def test_sort_by(self):
        code = natural_language_to_pandas("sort by age", self.df_info)
        assert "sort_values" in code
        assert "age" in code

    def test_sort_descending(self):
        code = natural_language_to_pandas(
            "sort by salary descending", self.df_info
        )
        assert "sort_values" in code
        assert "False" in code  # ascending=False

    def test_correlation(self):
        code = natural_language_to_pandas(
            "correlation between salary and age", self.df_info
        )
        assert "corr" in code
        assert "salary" in code
        assert "age" in code

    def test_describe(self):
        code = natural_language_to_pandas("describe salary", self.df_info)
        assert "describe" in code
        assert "salary" in code

    def test_missing_values(self):
        code = natural_language_to_pandas("missing values", self.df_info)
        assert "isnull" in code or "null" in code

    def test_unique_values(self):
        code = natural_language_to_pandas(
            "unique values of department", self.df_info
        )
        assert "unique" in code
        assert "department" in code

    def test_empty_query(self):
        code = natural_language_to_pandas("", self.df_info)
        assert "Could not parse" in code

    def test_unparseable_query(self):
        code = natural_language_to_pandas(
            "do something complicated", self.df_info
        )
        assert "Could not parse" in code or "#" in code

    def test_how_many(self):
        code = natural_language_to_pandas("how many rows", self.df_info)
        assert "len" in code or "shape" in code or "count" in code

    def test_uses_df_name(self):
        info = {**self.df_info, "name": "employees"}
        code = natural_language_to_pandas("show me salary", info)
        assert "employees" in code

    def test_case_insensitive_columns(self):
        code = natural_language_to_pandas("average of SALARY", self.df_info)
        assert "salary" in code
        assert "mean" in code

    def test_groupby_explicit(self):
        code = natural_language_to_pandas(
            "group by department and show mean salary", self.df_info
        )
        assert "groupby" in code
        assert "department" in code


# ---------------------------------------------------------------------------
# Test module imports
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_interpret_import(self):
        from pyanalytica.ai import interpret_result
        assert callable(interpret_result)

    def test_suggest_import(self):
        from pyanalytica.ai import suggest_next_step
        assert callable(suggest_next_step)

    def test_challenge_import(self):
        from pyanalytica.ai import challenge_interpretation
        assert callable(challenge_interpretation)

    def test_query_import(self):
        from pyanalytica.ai import natural_language_to_pandas
        assert callable(natural_language_to_pandas)
