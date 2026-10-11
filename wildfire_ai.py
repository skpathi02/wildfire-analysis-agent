"""Grounded natural-language wildfire analysis with LangSmith tracing."""

from __future__ import annotations

import os

import pandas as pd
from langsmith import traceable


class AIConfigurationError(RuntimeError):
    """Raised when the configured model provider cannot be used."""


def build_dataset_context(data: pd.DataFrame) -> str:
    """Create a compact, deterministic context from only the loaded rows."""
    if data.empty:
        return "The filtered dataset contains no rows."

    yearly = (
        data.groupby("year", as_index=False)[["wildfire_count", "acres_burned"]]
        .sum()
        .sort_values("year")
    )
    states = (
        data.groupby("state", as_index=False)[["wildfire_count", "acres_burned"]]
        .sum()
        .sort_values("acres_burned", ascending=False)
        .head(10)
    )
    months = (
        data.groupby(["month", "month_name"], as_index=False)[
            ["wildfire_count", "acres_burned"]
        ]
        .sum()
        .sort_values("wildfire_count", ascending=False)
    )
    seasons = data.groupby("season", as_index=False)[
        ["wildfire_count", "acres_burned"]
    ].sort_values("wildfire_count", ascending=False)

    return "\n".join(
        [
            f"Rows: {len(data)}",
            f"Year range: {int(data['year'].min())}–{int(data['year'].max())}",
            f"States represented: {data['state'].nunique()}",
            f"Total wildfires: {int(data['wildfire_count'].sum()):,}",
            f"Total acres burned: {data['acres_burned'].sum():,.0f}",
            "Yearly totals:\n" + yearly.to_csv(index=False),
            "Top states by acres burned:\n" + states.to_csv(index=False),
            "Monthly totals:\n" + months.to_csv(index=False),
            "Seasonal totals:\n" + seasons.to_csv(index=False),
        ]
    )


@traceable(name="wildfire-ai-analysis", run_type="chain")
def answer_dataset_question(question: str, data_context: str) -> str:
    """Answer a question using only the supplied aggregate dataset context."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AIConfigurationError(
            "OPENAI_API_KEY is not configured. Add it to .env or your environment "
            "to enable AI analysis."
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise AIConfigurationError(
            "AI dependencies are not installed. Install requirements-streamlit.txt."
        ) from exc

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You analyze a wildfire dashboard dataset. Answer only from the "
                    "DATASET CONTEXT below. Do not invent values, causes, locations, "
                    "or trends that are not supported by it. If the context cannot "
                    "answer the question, say so plainly. Distinguish synthetic "
                    "demonstration data from authentic data when relevant. Include "
                    "the relevant years, states, seasons, or totals used in your answer."
                )
            ),
            HumanMessage(
                content=(
                    f"QUESTION:\n{question}\n\nDATASET CONTEXT:\n{data_context}"
                )
            ),
        ]
    )
    return str(response.content)
