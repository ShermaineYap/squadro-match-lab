from __future__ import annotations

from datetime import date, datetime
from typing import Any
import math

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Recursively convert values into strict JSON-compatible Python objects."""
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, pd.Series):
        return to_json_safe(value.to_dict())

    if isinstance(value, pd.DataFrame):
        return to_json_safe(value.to_dict(orient="records"))

    if pd.isna(value):
        return None

    return str(value)
