from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import platform
import random

import joblib
import numpy as np
import pandas as pd
import sklearn


RANDOM_STATE = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any
    complexity_rank: int
    feature_columns: list[str]
    family: str
    imbalance_strategy: str = "none"


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects and non-finite values to strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def environment_metadata() -> dict[str, str]:
    metadata = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost
        metadata["xgboost"] = xgboost.__version__
    except Exception:
        metadata["xgboost"] = "not installed"
    try:
        import shap
        metadata["shap"] = shap.__version__
    except Exception:
        metadata["shap"] = "not installed"
    try:
        import imblearn
        metadata["imbalanced_learn"] = imblearn.__version__
    except Exception:
        metadata["imbalanced_learn"] = "not installed"
    metadata["joblib"] = joblib.__version__
    return metadata


def save_model_bundle(bundle: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


def make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_dense_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
