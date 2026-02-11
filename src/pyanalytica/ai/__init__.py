"""PyAnalytica AI module -- intelligent assistance for the analytics workbench.

This package provides four AI-powered capabilities:

1. **interpret** -- Generate plain-English interpretations of statistical
   and modeling results (means tests, correlations, regressions, etc.)

2. **suggest** -- Recommend the next analytical step based on the user's
   operation history and current dataset.

3. **challenge** -- Socratic questioning engine that challenges student
   interpretations without giving answers directly.

4. **query** -- Translate natural language questions into executable
   pandas code.

Each module works in two modes:
- **Rule-based** (always available): no API key or external packages needed.
- **LLM-enhanced** (optional): set your API key in ``~/.pyanalytica/profile.yaml``
  or the ``ANTHROPIC_API_KEY`` environment variable, and install the ``anthropic``
  package. Results will then be enriched with Claude.

Usage:
    from pyanalytica.ai import interpret_result, suggest_next_step
    from pyanalytica.ai import challenge_interpretation, natural_language_to_pandas
"""

from pyanalytica.ai.interpret import interpret_result
from pyanalytica.ai.suggest import suggest_next_step
from pyanalytica.ai.challenge import challenge_interpretation
from pyanalytica.ai.query import natural_language_to_pandas

__all__ = [
    "interpret_result",
    "suggest_next_step",
    "challenge_interpretation",
    "natural_language_to_pandas",
]
