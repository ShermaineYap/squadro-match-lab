from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_feature_frame(
    payload: dict[str, Any],
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    missing = [feature for feature in feature_columns if feature not in payload]
    if missing:
        raise ValueError(f"Missing required feature values: {missing}")

    row: dict[str, Any] = {}
    for feature in numeric_features:
        value = payload.get(feature)
        if value is None or value == "":
            row[feature] = np.nan
        else:
            try:
                row[feature] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{feature} must be numeric or blank, received {value!r}"
                ) from exc

    for feature in categorical_features:
        value = payload.get(feature)
        row[feature] = np.nan if value in (None, "") else value

    frame = pd.DataFrame([row])
    return frame[feature_columns]


def validate_category(
    feature: str,
    value: Any,
    allowed_values: list[Any],
    allow_missing: bool = False,
) -> Any:
    if value in (None, ""):
        if allow_missing:
            return np.nan
        raise ValueError(f"{feature} is required.")

    if value not in allowed_values:
        raise ValueError(
            f"Unsupported {feature} value {value!r}. "
            f"Choose one of: {allowed_values}"
        )
    return value
