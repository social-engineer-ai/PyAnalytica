"""PyAnalytica — A Python analytics workbench for teaching data science.

PyAnalytica is a package-first analytics workbench designed for business
school education. It works both as a Shiny web app and as a Python library
in Jupyter notebooks.

Quick start (web app)::

    pyanalytica          # CLI command
    # or
    python -m pyanalytica

Quick start (library)::

    from pyanalytica.data.load import load_bundled
    from pyanalytica.data.profile import profile_dataframe
    from pyanalytica.visualize.distribute import histogram

    df, code = load_bundled("tips")
    profile = profile_dataframe(df)
    fig, code = histogram(df, "total_bill")
"""

__version__ = "0.2.0"
