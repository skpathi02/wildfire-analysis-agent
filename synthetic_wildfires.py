"""Deterministic synthetic wildfire data for dashboard demonstrations."""

from __future__ import annotations

import math
import random

import pandas as pd

STATE_REGIONS = {
    "Alaska": "West",
    "Arizona": "Southwest",
    "California": "West",
    "Colorado": "Rocky Mountain",
    "Florida": "Southeast",
    "Georgia": "Southeast",
    "Idaho": "West",
    "Montana": "Rocky Mountain",
    "Nevada": "Southwest",
    "New Mexico": "Southwest",
    "North Carolina": "Southeast",
    "Oklahoma": "Great Plains",
    "Oregon": "West",
    "South Carolina": "Southeast",
    "Texas": "Great Plains",
    "Utah": "Rocky Mountain",
    "Virginia": "Southeast",
    "Washington": "West",
    "Wyoming": "Rocky Mountain",
}

MONTH_SEASON = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
    12: "Winter",
}


def generate_synthetic_data(
    start_year: int = 1975,
    end_year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """Return clearly synthetic monthly wildfire observations for 50 years."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    states = list(STATE_REGIONS.items())

    for year in range(start_year, end_year + 1):
        year_index = year - start_year
        long_term_factor = 1.0 + (year_index / max(1, end_year - start_year)) * 0.42
        for month in range(1, 13):
            seasonal_factor = {
                1: 0.30, 2: 0.25, 3: 0.45, 4: 0.65,
                5: 0.80, 6: 1.05, 7: 1.35, 8: 1.40,
                9: 1.15, 10: 0.75, 11: 0.45, 12: 0.28,
            }[month]
            for state, region in states:
                state_factor = 0.75 + (sum(ord(char) for char in state) % 19) / 25
                noise = rng.uniform(0.88, 1.14)
                count = max(
                    1,
                    round(12 * state_factor * seasonal_factor * long_term_factor * noise),
                )
                acres_per_fire = 180 + 95 * seasonal_factor + 2.5 * year_index
                acres_noise = rng.uniform(0.78, 1.26)
                acres = max(
                    count * 25,
                    round(count * acres_per_fire * acres_noise),
                )
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "state": state,
                        "region": region,
                        "wildfire_count": count,
                        "acres_burned": acres,
                    }
                )

    return pd.DataFrame(rows)


def synthetic_data_label() -> str:
    return (
        "Synthetic demonstration data: deterministic estimates created by this app "
        "for 1975–2024. Replace with an authentic dataset using the CSV uploader."
    )
