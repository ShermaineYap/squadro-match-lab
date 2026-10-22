from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import json
import numpy as np
import pandas as pd
import matplotlib

# Evidence generation runs in CI and other headless environments. Selecting the
# non-interactive backend before importing pyplot prevents macOS GUI backend
# aborts while producing the same saved figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from src.models.common import load_model_bundle, save_json


def _build_numeric_shap_view(
    background: pd.DataFrame,
    sample: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
):
    """Encode mixed raw inputs for SHAP while preserving a reversible map."""
    encoded_background = pd.DataFrame(index=background.index)
    encoded_sample = pd.DataFrame(index=sample.index)
    category_maps: dict[str, list] = {}

    categorical_set = set(categorical_features)
    for feature in feature_columns:
        if feature in categorical_set:
            combined = pd.concat(
                [background[feature], sample[feature]],
                ignore_index=True,
            )
            categories = list(pd.Series(combined.dropna().unique()))
            category_maps[feature] = categories
            lookup = {value: index for index, value in enumerate(categories)}
            encoded_background[feature] = (
                background[feature].map(lookup).fillna(-1).astype(float)
            )
            encoded_sample[feature] = (
                sample[feature].map(lookup).fillna(-1).astype(float)
            )
        else:
            encoded_background[feature] = pd.to_numeric(
                background[feature], errors="coerce"
            ).astype(float)
            encoded_sample[feature] = pd.to_numeric(
                sample[feature], errors="coerce"
            ).astype(float)

    return encoded_background, encoded_sample, category_maps


def _prediction_function(
    model: Any,
    feature_columns: list[str],
    categorical_features: list[str],
    category_maps: dict[str, list],
):
    categorical_set = set(categorical_features)

    def predict(values):
        if isinstance(values, pd.DataFrame):
            encoded = values.copy()
            encoded.columns = feature_columns
        else:
            encoded = pd.DataFrame(values, columns=feature_columns)

        decoded = pd.DataFrame(index=encoded.index)
        for feature in feature_columns:
            if feature in categorical_set:
                categories = category_maps[feature]

                def decode(value, categories=categories):
                    if pd.isna(value) or int(round(float(value))) < 0:
                        return np.nan
                    index = int(round(float(value)))
                    index = min(max(index, 0), len(categories) - 1)
                    return categories[index]

                decoded[feature] = encoded[feature].map(decode)
            else:
                decoded[feature] = pd.to_numeric(
                    encoded[feature], errors="coerce"
                )
        return model.predict_proba(decoded[feature_columns])

    return predict


def pipeline_level_shap(
    model_bundle: dict,
    background: pd.DataFrame,
    explain_rows: pd.DataFrame,
    target_class_index: int,
    maximum_rows: int = 25,
    random_state: int = 42,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Model-agnostic SHAP on raw user-facing features.

    Mixed categorical values are reversibly encoded only for SHAP's masker.
    The fitted pipeline still receives the original category values.
    """
    feature_columns = list(model_bundle["feature_columns"])
    categorical_features = list(model_bundle.get("categorical_features", []))
    base_model = model_bundle["base_model_for_explanation"]

    background = background[feature_columns].sample(
        n=min(80, len(background)),
        random_state=random_state,
    )
    sample = explain_rows[feature_columns].sample(
        n=min(maximum_rows, len(explain_rows)),
        random_state=random_state,
    )

    encoded_background, encoded_sample, category_maps = _build_numeric_shap_view(
        background,
        sample,
        feature_columns,
        categorical_features,
    )

    predict_function = _prediction_function(
        base_model,
        feature_columns,
        categorical_features,
        category_maps,
    )
    masker = shap.maskers.Independent(
        encoded_background,
        max_samples=min(80, len(encoded_background)),
    )
    explainer = shap.Explainer(
        predict_function,
        masker,
        algorithm="permutation",
        output_names=model_bundle["class_names"],
    )
    maximum_evaluations = max(2 * len(feature_columns) + 1, 51)
    explanation = explainer(
        encoded_sample,
        max_evals=maximum_evaluations,
    )

    if explanation.values.ndim == 3:
        values = explanation.values[:, :, target_class_index]
        base_values = explanation.base_values[:, target_class_index]
    else:
        values = explanation.values
        base_values = explanation.base_values

    selected = shap.Explanation(
        values=values,
        base_values=base_values,
        data=sample.to_numpy(),
        feature_names=feature_columns,
    )

    mean_absolute = np.abs(selected.values).mean(axis=0)
    global_importance = (
        pd.DataFrame(
            {
                "feature": selected.feature_names,
                "mean_absolute_shap": mean_absolute,
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    return selected, global_importance

def local_contributions(
    explanation: shap.Explanation,
    row_index: int = 0,
) -> pd.DataFrame:
    values = explanation.values[row_index]
    data = explanation.data[row_index]
    rows = []
    for feature, value, shap_value in zip(
        explanation.feature_names,
        data,
        values,
    ):
        rows.append(
            {
                "feature": feature,
                "feature_value": value,
                "shap_value": float(shap_value),
                "absolute_shap": abs(float(shap_value)),
                "direction": (
                    "toward target class"
                    if shap_value > 0
                    else "away from target class"
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_outputs(
    explanation: shap.Explanation,
    global_importance: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    target_label: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_importance.to_csv(
        output_dir / f"{prefix}_shap_global_importance.csv",
        index=False,
    )
    local = local_contributions(explanation, row_index=0)
    local.to_csv(
        output_dir / f"{prefix}_shap_local_example.csv",
        index=False,
    )

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title(f"{prefix.upper()} Global SHAP — {target_label}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{prefix}_shap_global_bar.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"{prefix.upper()} SHAP Summary — {target_label}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{prefix}_shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    shap.plots.waterfall(explanation[0], max_display=12, show=False)
    plt.title(f"{prefix.upper()} Local SHAP Example — {target_label}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{prefix}_shap_local_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    save_json(
        {
            "target_label": target_label,
            "base_value": float(np.asarray(explanation.base_values)[0]),
            "top_contributions": local.head(12).to_dict(orient="records"),
            "note": (
                "SHAP explains the underlying champion pipeline. Displayed "
                "probabilities may be calibrated separately. SHAP values are "
                "associations within the model, not causal effects."
            ),
        },
        output_dir / f"{prefix}_shap_local_example.json",
    )


def explain_saved_bundle(
    bundle_path: Path,
    background: pd.DataFrame,
    explain_rows: pd.DataFrame,
    target_class_index: int,
    output_dir: Path,
    prefix: str,
    maximum_rows: int = 25,
) -> dict:
    bundle = load_model_bundle(bundle_path)
    explanation, importance = pipeline_level_shap(
        bundle,
        background=background,
        explain_rows=explain_rows,
        target_class_index=target_class_index,
        maximum_rows=maximum_rows,
    )
    target_label = bundle["class_names"][target_class_index]
    save_shap_outputs(
        explanation,
        importance,
        output_dir,
        prefix,
        target_label,
    )
    return {
        "bundle": bundle,
        "explanation": explanation,
        "global_importance": importance,
        "local_contributions": local_contributions(explanation),
    }
