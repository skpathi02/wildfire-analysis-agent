"""Streamlit entry point for the historical wildfire dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from synthetic_wildfires import generate_synthetic_data, synthetic_data_label
from wildfire_ai import AIConfigurationError, answer_dataset_question, build_dataset_context
from wildfire_dashboard import (
    DataValidationError,
    geographic_chart,
    monthly_chart,
    read_uploaded_csv,
    seasonal_chart,
    state_summary,
    validate_and_prepare,
    yearly_acres_chart,
    yearly_count_chart,
)

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(
    page_title="US Wildfire Patterns",
    page_icon="🔥",
    layout="wide",
)

st.title("United States Wildfire Patterns-for GEN AI")
st.caption("Interactive analysis of 50 years of historical wildfire activity")

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader(
        "Upload a wildfire CSV",
        type=["csv"],
        help=(
            "Required columns: year, month, state, region, wildfire_count, "
            "acres_burned."
        ),
    )
    st.divider()
    st.markdown(
        "**CSV schema**  \n"
        "`year`, `month` (1–12), `state`, `region`, "
        "`wildfire_count`, `acres_burned`"
    )

if uploaded_file is None:
    data_source = "synthetic"
    data = validate_and_prepare(generate_synthetic_data())
    st.info(synthetic_data_label())
else:
    data_source = uploaded_file.name
    try:
        data = read_uploaded_csv(uploaded_file.getvalue())
        st.success(f"Loaded and validated `{uploaded_file.name}`.")
    except DataValidationError as exc:
        st.error(f"CSV validation failed: {exc}")
        st.stop()

if data.empty:
    st.warning("No rows are available for analysis.")
    st.stop()

st.sidebar.header("Filters")
year_min, year_max = int(data["year"].min()), int(data["year"].max())
if year_min == year_max:
    selected_years = (year_min, year_max)
    st.sidebar.caption(f"Only year available: {year_min}")
else:
    selected_years = st.sidebar.slider(
        "Year range", min_value=year_min, max_value=year_max,
        value=(year_min, year_max), step=1,
    )
all_states = sorted(data["state"].unique().tolist())
selected_states = st.sidebar.multiselect("States", all_states, default=all_states)

filtered = data[
    data["year"].between(selected_years[0], selected_years[1])
    & data["state"].isin(selected_states)
].copy()

if filtered.empty:
    st.warning("The selected filters return no rows. Choose at least one state and year range.")
    st.stop()

st.caption(f"Source: {data_source} · {len(filtered):,} filtered records")
metric_columns = st.columns(4)
metric_columns[0].metric("Wildfires", f"{filtered['wildfire_count'].sum():,.0f}")
metric_columns[1].metric("Acres burned", f"{filtered['acres_burned'].sum():,.0f}")
metric_columns[2].metric("States", f"{filtered['state'].nunique():,}")
metric_columns[3].metric("Years", f"{filtered['year'].nunique():,}")

st.subheader("Annual patterns")
annual_left, annual_right = st.columns(2)
with annual_left:
    st.plotly_chart(yearly_count_chart(filtered), use_container_width=True)
with annual_right:
    st.plotly_chart(yearly_acres_chart(filtered), use_container_width=True)

st.subheader("Monthly and seasonal trends")
trend_left, trend_right = st.columns(2)
with trend_left:
    st.plotly_chart(monthly_chart(filtered), use_container_width=True)
with trend_right:
    st.plotly_chart(seasonal_chart(filtered), use_container_width=True)

st.subheader("Geographic distribution")
st.plotly_chart(geographic_chart(filtered), use_container_width=True)
st.dataframe(
    state_summary(filtered), use_container_width=True, hide_index=True,
)

st.subheader("Ask about the loaded dataset")
st.caption(
    "AI answers are grounded in aggregates from the currently filtered rows. "
    "The AI feature requires OPENAI_API_KEY; LangSmith tracing uses the existing "
    "LANGSMITH_* environment configuration."
)
question = st.text_input(
    "Natural-language question",
    placeholder="Which state had the most acres burned in the selected period?",
)
if st.button("Analyze", type="primary", disabled=not question.strip()):
    with st.spinner("Analyzing the filtered dataset..."):
        try:
            answer = answer_dataset_question(
                question.strip(), build_dataset_context(filtered)
            )
            st.markdown(answer)
        except AIConfigurationError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "The AI analysis request failed. Check the model configuration and "
                "LangSmith/model-provider connectivity without sharing credentials."
            )

with st.expander("About the data"):
    if data_source == "synthetic":
        st.write(synthetic_data_label())
    else:
        st.write(
            "This dashboard is using the uploaded CSV. Verify its provenance and "
            "definitions before treating results as official statistics."
        )
