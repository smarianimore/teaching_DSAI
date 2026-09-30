# Exemplary Exploratory Data Analysis (EDA) pipeline

## Installation and execution

Run from this directory in a Python 3.11+ environment:

````
  python -m venv .venv
  Activate the environment using your operating system's command.
  python -m pip install -r requirements.txt
  python pipeline.py
  streamlit run app.py
````

Alternatively open `logistics_eda.ipynb` in JupyterLab and run all cells.
The notebook and script implement the same pipeline; use either one.

## Contents

````
  pipeline.py             End-to-end simulation, CSV/SQLite ingestion, audit,
                          cleaning, metrics, analysis, plots and HTML report.
  logistics_eda.ipynb     Notebook form of pipeline.py, with explanatory cells.
  app.py                  Interactive Streamlit dashboard with filters/download.
  outputs/                Precomputed core outputs from the supplied simulation.
  requirements.txt        Compatible version ranges, not a frozen environment.
  outputs/versions.json   Versions actually used to generate included outputs.
````

## Official documentation
  https://pandas.pydata.org/docs/user_guide/io.html
  https://pandas.pydata.org/docs/user_guide/missing_data.html
  https://pandas.pydata.org/docs/user_guide/merging.html
  https://pandas.pydata.org/docs/user_guide/groupby.html
  https://docs.scipy.org/doc/scipy/reference/stats.html
  https://www.statsmodels.org/stable/index.html
  https://scikit-learn.org/stable/user_guide.html
  https://matplotlib.org/stable/users/index.html
  https://seaborn.pydata.org/tutorial.html
  https://plotly.com/python/
  https://docs.streamlit.io/

