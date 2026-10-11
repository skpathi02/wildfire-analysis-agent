"""Data preparation, validation, and visualization helpers for the dashboard."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px

REQUIRED_COLUMNS = {
    "year",
    "month",
    "state",
    "region",
    "wildfire_count",
    "acres_burned",
}
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
SEASON_NAMES = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
    11: "Autumn", 12: "Winter",
}


class DataValidationError(ValueError):
    """Raised when an uploaded wildfire CSV does not match the app schema."""


def validate_and_prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the required wildfire data schema."""
    if data.empty:
        raise DataValidationError("The CSV contains no rows.")

    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing) + "."
        )

    prepared = data[list(REQUIRED_COLUMNS)].copy()
    for column in ("year", "month", "wildfire_count", "acres_burned"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if prepared[column].isna().any():
            raise DataValidationError(
                f"Column '{column}' must contain only numeric values."
            )

    for column in ("state", "region"):
        prepared[column] = prepared[column].astype("string").str.strip()
        if prepared[column].isna().any() or (prepared[column] == "").any():
            raise DataValidationError(f"Column '{column}' contains blank values.")

    if not (prepared["year"] % 1 == 0).all():
        raise DataValidationError("Column 'year' must contain whole years.")
    if not prepared["month"].between(1, 12).all():
        raise DataValidationError("Column 'month' must contain values from 1 to 12.")
    if (prepared["wildfire_count"] < 0).any():
        raise DataValidationError("Column 'wildfire_count' cannot contain negatives.")
    if (prepared["acres_burned"] < 0).any():
        raise DataValidationError("Column 'acres_burned' cannot contain negatives.")

    prepared["year"] = prepared["year"].astype(int)
    prepared["month"] = prepared["month"].astype(int)
    prepared["wildfire_count"] = prepared["wildfire_count"].astype(int)
    prepared["acres_burned"] = prepared["acres_burned"].astype(float)
    prepared["month_name"] = prepared["month"].map(MONTH_NAMES)
    prepared["season"] = prepared["month"].map(SEASON_NAMES)
    return prepared.sort_values(["year", "month", "state"]).reset_index(drop=True)


def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read and validate uploaded CSV bytes."""
    try:
        data = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise DataValidationError(f"Unable to read the CSV: {exc}") from exc
    return validate_and_prepare(data)


def yearly_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("year", as_index=False)[["wildfire_count", "acres_burned"]]
        .sum()
        .sort_values("year")
    )


def monthly_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = data.groupby(["month", "month_name"], as_index=False)[
        ["wildfire_count", "acres_burned"]
    ].sum()
    return summary.sort_values("month")


def seasonal_summary(data: pd.DataFrame) -> pd.DataFrame:
    order = ["Winter", "Spring", "Summer", "Autumn"]
    summary = data.groupby("season", as_index=False)[
        ["wildfire_count", "acres_burned"]
    ].sum()
    summary["season"] = pd.Categorical(summary["season"], categories=order, ordered=True)
    return summary.sort_values("season")


def state_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["state", "region"], as_index=False)[
            ["wildfire_count", "acres_burned"]
        ]
        .sum()
        .sort_values("acres_burned", ascending=False)
    )


def yearly_count_chart(data: pd.DataFrame):
    return px.line(
        yearly_summary(data), x="year", y="wildfire_count",
        markers=True, title="Wildfire counts by year",
        labels={"wildfire_count": "Wildfires", "year": "Year"},
    )


def yearly_acres_chart(data: pd.DataFrame):
    return px.area(
        yearly_summary(data), x="year", y="acres_burned",
        title="Total acres burned by year",
        labels={"acres_burned": "Acres burned", "year": "Year"},
    )


def monthly_chart(data: pd.DataFrame):
    return px.bar(
        monthly_summary(data), x="month_name", y="wildfire_count",
        title="Monthly wildfire pattern",
        labels={"month_name": "Month", "wildfire_count": "Wildfires"},
    )


def seasonal_chart(data: pd.DataFrame):
    return px.bar(
        seasonal_summary(data), x="season", y="acres_burned",
        color="season", title="Acres burned by season",
        labels={"season": "Season", "acres_burned": "Acres burned"},
    )


def geographic_chart(data: pd.DataFrame):
    summary = state_summary(data)
    return px.bar(
        summary.head(20), x="acres_burned", y="state", color="region",
        orientation="h", title="Geographic distribution by state",
        labels={"acres_burned": "Acres burned", "state": "State"},
    ).update_layout(yaxis={"categoryorder": "total ascending"})
