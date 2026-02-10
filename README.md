# PyAnalytica

A Python analytics workbench for teaching data science. Built on Shiny for Python.

## Features

- Interactive data exploration and visualization
- Guided statistical analysis workflows
- Machine learning model building and evaluation
- AI-powered data insights (optional Anthropic integration)
- Homework assignment framework for instructors
- Automated report generation
- Themeable UI with a clean, modern design

## Installation

```bash
pip install pyanalytica
```

Optional extras:

```bash
pip install pyanalytica[ai]      # Anthropic AI integration
pip install pyanalytica[report]  # Report generation
pip install pyanalytica[dev]     # Development and testing
```

## Quick Start

Launch the interactive workbench:

```bash
pyanalytica
```

Or import directly in a Jupyter notebook:

```python
import pyanalytica as pa

# Load a dataset
df = pa.data.load("iris")

# Quick summary
pa.explore.summary(df)

# Visualize
pa.visualize.scatter(df, x="sepal_length", y="sepal_width", color="species")
```

## License

MIT License. See [LICENSE](LICENSE) for details.
