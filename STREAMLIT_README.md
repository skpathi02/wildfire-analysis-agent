# Wildfire dashboard

`streamlit_app.py` is a separate Streamlit application for exploring 50 years of US wildfire patterns. It does not replace or modify the Managed Deep Agent entry points in `agent.py`, `identity.py`, `instructions.md`, or `pyproject.toml`.

## Features

- Deterministic synthetic demonstration data for 1975–2024, clearly labeled as synthetic
- CSV upload with schema and value validation
- Year-range and state filters
- Interactive Plotly views for annual counts, annual acres burned, monthly trends, seasonal trends, and state distribution
- Natural-language questions about the currently filtered dataset
- LangSmith tracing for AI analysis calls through `@traceable`

## CSV format

Uploaded files must contain these columns:

```text
year,month,state,region,wildfire_count,acres_burned
```

`month` must be an integer from 1 through 12. Counts and acres must be numeric and nonnegative. The dashboard does not infer missing columns or silently repair invalid values.

## Local setup

From this directory, create an environment and install the separate dashboard dependencies:

```bash
python3 -m venv .venv-streamlit
. .venv-streamlit/bin/activate
python -m pip install -r requirements-streamlit.txt
```

On Windows PowerShell, activate with:

```powershell
.venv-streamlit\Scripts\Activate.ps1
```

Run the app:

```bash
streamlit run streamlit_app.py
```

## Environment configuration

Put credentials in the existing local `.env` file or configure them in the shell. Do not commit or display them. The AI question feature needs an OpenAI credential:

```text
OPENAI_API_KEY=...
```

LangSmith tracing uses the existing LangSmith configuration. These values are examples of variable names only:

```text
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=wildfire-analysis-agent
```

`OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`. The app reads environment variables without rendering their values. If `OPENAI_API_KEY` is absent, the dashboard and charts still work and the AI panel reports that configuration is incomplete.

The AI prompt receives only aggregates generated from the currently filtered dataframe. It is instructed not to infer unsupported facts; synthetic data is identified as synthetic in the response context.
