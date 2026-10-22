
"""Shared final data-understanding and preprocessing workflow."""

from __future__ import annotations

from pathlib import Path


def run_data_workflow(project_root: str | Path, show_outputs: bool = True) -> dict:
    """Run the complete two-stream workflow and return the main result objects."""
    # %% [markdown]
    # # Final FYP Data Understanding and Data Preprocessing
    #
    # **Project:** Predicting Injury Risk and Performance in Badminton Players Using Machine Learning Techniques
    #
    # This notebook implements the final **CRISP-DM Data Understanding and Data Preparation** workflow for:
    #
    # 1. **BWF match-performance data, 2021–2025**
    # 2. **Multimodal sports injury data**
    #
    # The notebook keeps the Investigation Report's report-oriented structure while applying stricter leakage controls required by the final prediction tasks.
    #
    # ## Final methodological decisions
    #
    # ### BWF stream
    #
    # - The task is a genuine **pre-match** winner prediction problem.
    # - Game scores, number of games, score margins and the winning team name are used only for data-quality auditing. They are not predictors.
    # - Player IDs are retained temporarily as historical feature-engineering keys, but excluded from the model.
    # - Exact within-day match order is unavailable for most records. All matches on the same date therefore use history from strictly earlier dates.
    # - 2021 is a historical warm-up period, 2022–2023 are training, 2024 is validation and 2025 is the final untouched test period.
    #
    # ### Injury stream
    #
    # - The task is described as athlete-session injury-risk classification or screening because the dataset does not establish clinical causation.
    # - Athlete IDs are retained for grouped splitting but excluded from predictors.
    # - The final test set contains athletes not seen in development.
    # - Missing-value imputation, scaling, encoding, optional capping and resampling are fitted inside later training folds only.
    # - Statistical outliers are documented rather than automatically removed.
    #
    # ## Main outputs
    #
    # - formal variable dictionaries;
    # - data-quality and missing-value reports;
    # - report-ready exploratory visualisations;
    # - athlete-disjoint injury development/test files;
    # - validated BWF data with quality flags;
    # - leakage-safe, date-batched BWF historical features;
    # - chronological BWF split labels;
    # - feature manifests and preprocessing-pipeline templates.

    # %%

    from pathlib import Path
    from collections import defaultdict, deque
    import json
    import platform
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from IPython.display import display as _ipython_display

    def display(value):
        if show_outputs:
            _ipython_display(value)
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    BASE_DIR = Path(project_root).resolve()
    RAW_DIR = BASE_DIR / "data" / "raw"
    OUTPUT_DIR = BASE_DIR / "outputs"
    INJURY_OUT = OUTPUT_DIR / "injury"
    BWF_OUT = OUTPUT_DIR / "bwf"

    for folder in [OUTPUT_DIR, INJURY_OUT, BWF_OUT]:
        folder.mkdir(parents=True, exist_ok=True)

    INJURY_PATH = RAW_DIR / "multimodal_sports_injury_dataset.csv"
    BWF_PATH = RAW_DIR / "bwf_matches_2021_2025.csv"

    assert INJURY_PATH.exists(), f"Missing file: {INJURY_PATH}"
    assert BWF_PATH.exists(), f"Missing file: {BWF_PATH}"

    print("Python:", platform.python_version())
    print("Pandas:", pd.__version__)
    print("Base directory:", BASE_DIR)

    # %% [markdown]
    # ## 1. Load the raw datasets

    # %%

    injury_raw = pd.read_csv(INJURY_PATH)
    bwf_raw = pd.read_csv(BWF_PATH)

    dataset_overview = pd.DataFrame({
        "dataset": ["Multimodal sports injury", "BWF matches 2021–2025"],
        "rows": [len(injury_raw), len(bwf_raw)],
        "columns": [injury_raw.shape[1], bwf_raw.shape[1]],
        "exact_duplicate_rows": [injury_raw.duplicated().sum(), bwf_raw.duplicated().sum()],
    })

    display(dataset_overview)
    display(injury_raw.head(3))
    display(bwf_raw.head(3))

    # %%

    def column_profile(df):
        profile = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100),
            "unique": df.nunique(dropna=True),
        })
        return profile.sort_values(["missing_pct", "unique"], ascending=[False, False])

    def plot_value_counts(series, title, xlabel="", ylabel="Count", max_categories=None):
        counts = series.value_counts(dropna=False)
        if max_categories is not None:
            counts = counts.head(max_categories)
        plt.figure(figsize=(9, 4.5))
        counts.plot(kind="bar")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show() if show_outputs else plt.close()

    def safe_difference(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float(a - b)

    # %% [markdown]
    # # Part A — Multimodal Sports Injury Dataset

    # %% [markdown]
    # ## Injury variable dictionary

    # %%

    injury_descriptions = {
        "athlete_id": "Unique athlete identifier.",
        "heart_rate": "Recorded athlete heart rate.",
        "body_temperature": "Recorded body temperature.",
        "hydration_level": "Hydration measurement or score.",
        "sleep_quality": "Sleep quality score.",
        "recovery_score": "Athlete recovery score.",
        "stress_level": "Recorded stress score.",
        "muscle_activity": "Muscle-activity measurement.",
        "joint_angles": "Joint-angle measurement.",
        "gait_speed": "Recorded gait speed.",
        "cadence": "Recorded movement cadence.",
        "step_count": "Number of recorded steps.",
        "jump_height": "Recorded jump-height measurement.",
        "ground_reaction_force": "Ground-reaction-force measurement.",
        "range_of_motion": "Range-of-motion measurement.",
        "ambient_temperature": "Environmental temperature.",
        "humidity": "Environmental humidity.",
        "altitude": "Environmental altitude.",
        "playing_surface": "Coded playing-surface category.",
        "training_intensity": "Training-intensity measurement.",
        "training_duration": "Duration of the training session.",
        "training_load": "Training-load measurement.",
        "fatigue_index": "Fatigue measurement or score.",
        "injury_occurred": "Provider-defined risk target: 0 Healthy, 1 Low Risk, 2 High Risk/Injured.",
        "session_id": "Session identifier used for audit and grouping checks.",
        "sport_type": "Sport category.",
        "gender": "Recorded gender category.",
        "age": "Athlete age.",
        "bmi": "Body mass index.",
    }

    injury_dictionary_rows = []
    for column in injury_raw.columns:
        if column in {"athlete_id", "session_id"}:
            role = "Identifier / grouping key"
            action = "Retain for audit and splitting; exclude from predictors"
            model_allowed = "No"
        elif column == "injury_occurred":
            role = "Target"
            action = "Keep as three-class target"
            model_allowed = "Target only"
        else:
            role = "Candidate predictor"
            action = "Retain; preprocess inside training folds"
            model_allowed = "Yes"

        injury_dictionary_rows.append({
            "variable": column,
            "description": injury_descriptions.get(column, ""),
            "raw_dtype": str(injury_raw[column].dtype),
            "unique_values": injury_raw[column].nunique(dropna=True),
            "missing": injury_raw[column].isna().sum(),
            "missing_pct": injury_raw[column].isna().mean() * 100,
            "role": role,
            "preprocessing_action": action,
            "allowed_in_model": model_allowed,
        })

    injury_data_dictionary = pd.DataFrame(injury_dictionary_rows)
    display(injury_data_dictionary)
    injury_data_dictionary.to_csv(
        INJURY_OUT / "injury_variable_dictionary.csv", index=False
    )

    # %% [markdown]
    # ## 2. Injury data structure and quality

    # %%

    injury_profile = column_profile(injury_raw)
    display(injury_profile)

    injury_key_checks = pd.DataFrame({
        "check": [
            "Rows",
            "Columns",
            "Unique athletes",
            "Unique session IDs",
            "Exact duplicate rows",
            "Duplicate athlete-session pairs",
            "Missing target values",
            "Unexpected target labels",
        ],
        "result": [
            len(injury_raw),
            injury_raw.shape[1],
            injury_raw["athlete_id"].nunique(),
            injury_raw["session_id"].nunique(),
            injury_raw.duplicated().sum(),
            injury_raw.duplicated(["athlete_id", "session_id"]).sum(),
            injury_raw["injury_occurred"].isna().sum(),
            (~injury_raw["injury_occurred"].isin([0, 1, 2])).sum(),
        ],
    })
    display(injury_key_checks)

    # %%

    injury_missing = injury_raw.isna().sum()
    injury_missing = injury_missing[injury_missing > 0].sort_values(ascending=False)

    display(pd.DataFrame({
        "missing": injury_missing,
        "missing_pct": injury_missing.div(len(injury_raw)).mul(100),
    }))

    plt.figure(figsize=(9, 4.5))
    injury_missing.plot(kind="bar")
    plt.title("Injury dataset: missing values by variable")
    plt.xlabel("Variable")
    plt.ylabel("Missing rows")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    # %% [markdown]
    # ## 3. Injury target and categorical distributions

    # %%

    injury_label_map = {0: "Healthy", 1: "Low Risk", 2: "High Risk/Injured"}

    injury_target_summary = (
        injury_raw["injury_occurred"]
        .value_counts()
        .sort_index()
        .rename(index=injury_label_map)
        .to_frame("records")
    )
    injury_target_summary["percentage"] = injury_target_summary["records"].div(len(injury_raw)).mul(100)

    display(injury_target_summary)
    plot_value_counts(
        injury_raw["injury_occurred"].map(injury_label_map),
        "Injury outcome distribution",
        xlabel="Outcome"
    )

    display(injury_raw["sport_type"].value_counts(dropna=False).to_frame("records"))
    display(injury_raw["gender"].value_counts(dropna=False).to_frame("records"))
    display(pd.crosstab(injury_raw["sport_type"], injury_raw["injury_occurred"], normalize="index").rename(columns=injury_label_map))
    display(pd.crosstab(injury_raw["gender"], injury_raw["injury_occurred"], normalize="index").rename(columns=injury_label_map))

    # %% [markdown]
    # ## 4. Repeated-athlete and session structure

    # %%

    rows_per_athlete = injury_raw.groupby("athlete_id").size()
    athletes_per_session = injury_raw.groupby("session_id")["athlete_id"].nunique()

    display(rows_per_athlete.describe().to_frame("rows_per_athlete"))
    display(athletes_per_session.describe().to_frame("athletes_per_session"))

    athlete_static_checks = []
    for column in ["sport_type", "gender", "age", "bmi"]:
        counts = injury_raw.groupby("athlete_id")[column].nunique(dropna=False)
        athlete_static_checks.append({
            "column": column,
            "athletes_with_multiple_values": int((counts > 1).sum()),
            "maximum_values_per_athlete": int(counts.max()),
        })

    display(pd.DataFrame(athlete_static_checks))

    print(
        "Interpretation: the dataset contains repeated observations per athlete. "
        "Therefore, athlete_id must be used as a grouping variable during validation, "
        "but it should not be used as a predictor."
    )

    # %% [markdown]
    # ## 5. Injury numeric summaries and exploratory relationships

    # %%

    injury_numeric_summary = injury_raw.select_dtypes(include=np.number).describe().T
    display(injury_numeric_summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    numeric_for_correlation = [
        c for c in injury_raw.select_dtypes(include=np.number).columns
        if c not in ["athlete_id", "session_id", "injury_occurred"]
    ]

    spearman_with_target = (
        injury_raw[numeric_for_correlation + ["injury_occurred"]]
        .corr(method="spearman")["injury_occurred"]
        .drop("injury_occurred")
        .sort_values()
    )

    display(spearman_with_target.to_frame("spearman_correlation_with_injury_label"))

    plt.figure(figsize=(9, 7))
    spearman_with_target.plot(kind="barh")
    plt.title("Exploratory Spearman correlations with injury label")
    plt.xlabel("Correlation")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    print(
        "These correlations are exploratory associations only. "
        "They do not establish causation and must not be used as clinical conclusions."
    )

    # %% [markdown]
    # ## Injury outlier, range and multicollinearity analysis

    # %%

    injury_numeric_candidates = [
        c for c in injury_raw.select_dtypes(include=np.number).columns
        if c not in ["athlete_id", "session_id", "injury_occurred", "playing_surface"]
    ]

    outlier_rows = []
    for column in injury_numeric_candidates:
        values = injury_raw[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_count = ((values < lower_fence) | (values > upper_fence)).sum()

        outlier_rows.append({
            "variable": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_count": int(outlier_count),
            "outlier_pct": outlier_count / len(values) * 100,
        })

    injury_outlier_table = (
        pd.DataFrame(outlier_rows)
        .sort_values(["outlier_count", "variable"], ascending=[False, True])
    )
    display(injury_outlier_table)
    injury_outlier_table.to_csv(
        INJURY_OUT / "injury_iqr_outlier_report.csv", index=False
    )

    range_checks = [
        ("heart_rate <= 0", injury_raw["heart_rate"].le(0)),
        ("body_temperature <= 0", injury_raw["body_temperature"].le(0)),
        ("hydration_level outside 0–100", injury_raw["hydration_level"].notna() & ~injury_raw["hydration_level"].between(0, 100)),
        ("sleep_quality outside 0–10", injury_raw["sleep_quality"].notna() & ~injury_raw["sleep_quality"].between(0, 10)),
        ("recovery_score outside 0–100", injury_raw["recovery_score"].notna() & ~injury_raw["recovery_score"].between(0, 100)),
        ("stress_level outside 0–1", injury_raw["stress_level"].notna() & ~injury_raw["stress_level"].between(0, 1)),
        ("humidity outside 0–100", injury_raw["humidity"].notna() & ~injury_raw["humidity"].between(0, 100)),
        ("training_duration <= 0", injury_raw["training_duration"].le(0)),
        ("age <= 0", injury_raw["age"].le(0)),
        ("bmi <= 0", injury_raw["bmi"].le(0)),
        ("playing_surface outside provider codes 0–4", ~injury_raw["playing_surface"].isin([0, 1, 2, 3, 4])),
        ("injury target outside 0,1,2", ~injury_raw["injury_occurred"].isin([0, 1, 2])),
    ]

    range_report = pd.DataFrame([
        {
            "check": description,
            "flagged_rows": int(mask.fillna(False).sum()),
        }
        for description, mask in range_checks
    ])
    display(range_report)
    range_report.to_csv(INJURY_OUT / "injury_range_check_report.csv", index=False)

    print(
        "Policy: statistical outliers are documented, not automatically removed or capped. "
        "Extreme observations may represent important high-risk sessions. "
        "Any optional capping must be fitted inside later training folds and compared by validation."
    )

    # %%

    correlation_columns = injury_numeric_candidates + ["injury_occurred"]
    injury_corr = injury_raw[correlation_columns].corr(method="spearman")

    plt.figure(figsize=(12, 10))
    image = plt.imshow(injury_corr, aspect="auto", interpolation="nearest")
    plt.colorbar(image, label="Spearman correlation")
    plt.title("Injury dataset: Spearman correlation heatmap")
    plt.xticks(
        range(len(injury_corr.columns)),
        injury_corr.columns,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        range(len(injury_corr.index)),
        injury_corr.index,
        fontsize=7,
    )
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    high_correlation_pairs = []
    for i, left in enumerate(injury_corr.columns):
        for j in range(i + 1, len(injury_corr.columns)):
            right = injury_corr.columns[j]
            value = injury_corr.loc[left, right]
            if abs(value) >= 0.70:
                high_correlation_pairs.append({
                    "variable_1": left,
                    "variable_2": right,
                    "spearman_correlation": value,
                })

    high_correlation_pairs = pd.DataFrame(high_correlation_pairs)
    display(high_correlation_pairs)
    high_correlation_pairs.to_csv(
        INJURY_OUT / "injury_high_correlation_pairs.csv", index=False
    )

    # %% [markdown]
    # ## Injury report-ready visualisations

    # %%

    key_histogram_features = [
        "recovery_score",
        "sleep_quality",
        "fatigue_index",
        "training_load",
    ]

    for column in key_histogram_features:
        plt.figure(figsize=(8, 4.5))
        injury_raw[column].dropna().plot(kind="hist", bins=30)
        plt.title(f"Injury dataset: distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Records")
        plt.tight_layout()
        plt.show() if show_outputs else plt.close()

    # %%

    boxplot_features = [
        "recovery_score",
        "sleep_quality",
        "fatigue_index",
        "training_load",
    ]

    for column in boxplot_features:
        groups = [
            injury_raw.loc[injury_raw["injury_occurred"] == label, column].dropna()
            for label in [0, 1, 2]
        ]
        plt.figure(figsize=(8, 4.5))
        plt.boxplot(groups, tick_labels=["Healthy", "Low Risk", "High Risk/Injured"])
        plt.title(f"{column} by injury class")
        plt.xlabel("Injury class")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show() if show_outputs else plt.close()

    sample_for_scatter = injury_raw[
        ["training_load", "fatigue_index", "injury_occurred"]
    ].dropna().sample(
        n=min(4000, injury_raw[["training_load", "fatigue_index"]].dropna().shape[0]),
        random_state=42,
    )

    plt.figure(figsize=(8, 5))
    for label, label_name in injury_label_map.items():
        subset = sample_for_scatter[sample_for_scatter["injury_occurred"] == label]
        plt.scatter(
            subset["training_load"],
            subset["fatigue_index"],
            s=10,
            alpha=0.35,
            label=label_name,
        )
    plt.title("Training load versus fatigue index")
    plt.xlabel("Training load")
    plt.ylabel("Fatigue index")
    plt.legend()
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    # %% [markdown]
    # ## 6. Injury deterministic cleaning

    # %%

    injury_clean = injury_raw.copy()

    # Standardise text without inventing new values.
    for column in injury_clean.select_dtypes(include="object").columns:
        injury_clean[column] = injury_clean[column].astype("string").str.strip()

    # Remove exact duplicate rows only. None were found in the uploaded file,
    # but this keeps the procedure reusable.
    injury_clean = injury_clean.drop_duplicates().reset_index(drop=True)

    # Validate keys and target.
    if injury_clean[["athlete_id", "session_id"]].isna().any().any():
        raise ValueError("athlete_id and session_id must not be missing.")

    if not set(injury_clean["injury_occurred"].dropna().unique()).issubset({0, 1, 2}):
        raise ValueError("Unexpected injury target values were found.")

    injury_clean["athlete_id"] = injury_clean["athlete_id"].astype("int64")
    injury_clean["session_id"] = injury_clean["session_id"].astype("int64")
    injury_clean["injury_occurred"] = injury_clean["injury_occurred"].astype("int64")
    injury_clean["injury_label"] = injury_clean["injury_occurred"].map(injury_label_map)

    # playing_surface is a coded category rather than a continuous measurement.
    injury_clean["playing_surface"] = injury_clean["playing_surface"].astype("int64").astype("string")

    # Preserve missing values for leakage-safe imputation later.
    feature_columns_excluding_ids = [
        c for c in injury_clean.columns
        if c not in ["athlete_id", "session_id", "injury_occurred", "injury_label"]
    ]
    injury_clean["missing_feature_count"] = injury_clean[feature_columns_excluding_ids].isna().sum(axis=1)

    print("Rows after deterministic cleaning:", len(injury_clean))
    print("Rows with at least one missing predictor:", (injury_clean["missing_feature_count"] > 0).sum())
    display(injury_clean.head())

    # %% [markdown]
    # ## 7. Athlete-disjoint injury split

    # %%
    # Select an athlete-disjoint final test set from multiple valid candidate splits.
    # This preserves zero athlete overlap while choosing a split whose class, sport
    # and gender distributions are reasonably close to the full dataset.

    def normalized_distribution(frame, column):
        return frame[column].value_counts(normalize=True, dropna=False)

    def distribution_distance(full_frame, subset_frame, column):
        full_dist = normalized_distribution(full_frame, column)
        subset_dist = normalized_distribution(subset_frame, column)
        categories = full_dist.index.union(subset_dist.index)
        return float(
            (
                full_dist.reindex(categories, fill_value=0)
                - subset_dist.reindex(categories, fill_value=0)
            ).abs().sum()
        )

    candidate_splits = []

    for seed in range(300):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        development_index, test_index = next(
            splitter.split(
                injury_clean,
                y=injury_clean["injury_occurred"],
                groups=injury_clean["athlete_id"],
            )
        )

        development_candidate = injury_clean.iloc[development_index]
        test_candidate = injury_clean.iloc[test_index]

        # Require all target classes in both partitions.
        if development_candidate["injury_occurred"].nunique() < 3:
            continue
        if test_candidate["injury_occurred"].nunique() < 3:
            continue

        score = (
            3.0 * distribution_distance(
                injury_clean, test_candidate, "injury_occurred"
            )
            + distribution_distance(
                injury_clean, test_candidate, "sport_type"
            )
            + distribution_distance(
                injury_clean, test_candidate, "gender"
            )
            + abs(len(test_candidate) / len(injury_clean) - 0.20)
        )

        candidate_splits.append({
            "seed": seed,
            "score": score,
            "development_index": development_index,
            "test_index": test_index,
            "test_records": len(test_candidate),
            "test_athletes": test_candidate["athlete_id"].nunique(),
        })

    best_candidate = min(candidate_splits, key=lambda row: row["score"])
    selected_split_seed = best_candidate["seed"]
    development_index = best_candidate["development_index"]
    test_index = best_candidate["test_index"]

    injury_clean["split"] = ""
    injury_clean.loc[development_index, "split"] = "development"
    injury_clean.loc[test_index, "split"] = "test"

    development_athletes = set(
        injury_clean.loc[
            injury_clean["split"] == "development", "athlete_id"
        ]
    )
    test_athletes = set(
        injury_clean.loc[
            injury_clean["split"] == "test", "athlete_id"
        ]
    )

    assert development_athletes.isdisjoint(test_athletes)

    split_summary = injury_clean.groupby("split").agg(
        records=("athlete_id", "size"),
        athletes=("athlete_id", "nunique"),
    )
    display(split_summary)

    split_class_distribution = pd.crosstab(
        injury_clean["split"],
        injury_clean["injury_label"],
        normalize="index",
    )
    display(split_class_distribution)

    split_sport_distribution = pd.crosstab(
        injury_clean["split"],
        injury_clean["sport_type"],
        normalize="index",
    )
    display(split_sport_distribution)

    split_gender_distribution = pd.crosstab(
        injury_clean["split"],
        injury_clean["gender"],
        normalize="index",
    )
    display(split_gender_distribution)

    candidate_split_report = pd.DataFrame([
        {
            "seed": row["seed"],
            "distribution_score": row["score"],
            "test_records": row["test_records"],
            "test_athletes": row["test_athletes"],
        }
        for row in candidate_splits
    ]).sort_values("distribution_score")

    display(candidate_split_report.head(10))
    candidate_split_report.to_csv(
        INJURY_OUT / "injury_candidate_group_splits.csv", index=False
    )

    print("Selected split seed:", selected_split_seed)
    print("Athlete overlap:", len(development_athletes.intersection(test_athletes)))

    # %% [markdown]
    # ## 8. Injury feature manifest and preprocessing pipeline template

    # %%
    injury_identifier_columns = ["athlete_id", "session_id"]
    injury_target_column = "injury_occurred"

    injury_categorical_features = [
        "playing_surface",
        "sport_type",
        "gender",
    ]

    injury_numeric_features = [
        c for c in injury_clean.columns
        if c not in (
            injury_identifier_columns
            + injury_categorical_features
            + [
                "injury_occurred",
                "injury_label",
                "split",
                "missing_feature_count",
            ]
        )
        and pd.api.types.is_numeric_dtype(injury_clean[c])
    ]

    injury_feature_manifest = {
        "identifiers_not_used_as_predictors": injury_identifier_columns,
        "target": injury_target_column,
        "target_mapping": injury_label_map,
        "numeric_features": injury_numeric_features,
        "categorical_features": injury_categorical_features,
        "grouping_variable": "athlete_id",
        "selected_group_split_seed": int(selected_split_seed),
        "final_test_policy": (
            "Athlete-disjoint final test set; grouped cross-validation "
            "inside development data."
        ),
        "missing_value_policy": (
            "Preserve missing values in cleaned data. "
            "Fit imputers inside each training fold."
        ),
        "outlier_policy": (
            "Document statistical outliers. Do not automatically remove or cap them. "
            "Any optional capping must be fitted inside training folds and validated."
        ),
        "resampling_policy": (
            "Compare no resampling, class weighting and SMOTENC. "
            "Resampling must occur inside training folds only."
        ),
    }

    with open(
        INJURY_OUT / "injury_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(injury_feature_manifest, file, indent=2)

    display(pd.DataFrame({
        "numeric_features": pd.Series(injury_numeric_features),
        "categorical_features": pd.Series(injury_categorical_features),
    }))

    def build_injury_preprocessor():
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("numeric", numeric_pipeline, injury_numeric_features),
            ("categorical", categorical_pipeline, injury_categorical_features),
        ])

    injury_preprocessor = build_injury_preprocessor()
    print(injury_preprocessor)

    print(
        "\nImportant: keep this preprocessor inside the later model pipeline. "
        "Do not fit it once on the complete dataset before grouped cross-validation."
    )

    # %%

    injury_clean.to_csv(INJURY_OUT / "injury_validated_clean.csv", index=False)
    injury_clean.loc[injury_clean["split"] == "development"].to_csv(
        INJURY_OUT / "injury_development.csv", index=False
    )
    injury_clean.loc[injury_clean["split"] == "test"].to_csv(
        INJURY_OUT / "injury_final_test.csv", index=False
    )

    injury_clean[["athlete_id", "split"]].drop_duplicates().sort_values("athlete_id").to_csv(
        INJURY_OUT / "injury_athlete_split_assignment.csv", index=False
    )

    print("Injury outputs saved to:", INJURY_OUT)

    # %% [markdown]
    # # Part B — BWF Match-Performance Dataset, 2021–2025

    # %% [markdown]
    # ## BWF variable dictionary

    # %%

    bwf_descriptions = {
        "match_id": "Unique match identifier.",
        "date": "Match date.",
        "start_datetime_utc": "Available source datetime; usually date-only at midnight.",
        "discipline": "MS, WS, MD, WD or XD.",
        "tournament_id": "Tournament identifier.",
        "tournament": "Tournament name.",
        "tier": "BWF World Tour tier.",
        "round": "Tournament round.",
        "is_qualification": "Qualification-stage indicator.",
        "host_location": "Tournament host location.",
        "team1": "Team 1 display name.",
        "team2": "Team 2 display name.",
        "team1_player1_id": "Team 1 first player identifier.",
        "team1_player1": "Team 1 first player name.",
        "team1_player1_country": "Team 1 first player country code.",
        "team1_player2_id": "Team 1 second player identifier for doubles.",
        "team1_player2": "Team 1 second player name for doubles.",
        "team1_player2_country": "Team 1 second player country code for doubles.",
        "team2_player1_id": "Team 2 first player identifier.",
        "team2_player1": "Team 2 first player name.",
        "team2_player1_country": "Team 2 first player country code.",
        "team2_player2_id": "Team 2 second player identifier for doubles.",
        "team2_player2": "Team 2 second player name for doubles.",
        "team2_player2_country": "Team 2 second player country code for doubles.",
        "winner": "Target side: 1 Team 1, 2 Team 2.",
        "winner_team": "Post-match winning-team name.",
        "number_of_games": "Post-match number of games.",
        "game1_team1": "Post-match Team 1 Game 1 score.",
        "game1_team2": "Post-match Team 2 Game 1 score.",
        "game2_team1": "Post-match Team 1 Game 2 score.",
        "game2_team2": "Post-match Team 2 Game 2 score.",
        "game3_team1": "Post-match Team 1 Game 3 score.",
        "game3_team2": "Post-match Team 2 Game 3 score.",
        "score": "Post-match raw score string.",
        "source_endpoint": "Source API endpoint retained for traceability.",
    }

    bwf_dictionary_rows = []
    post_match_set = {
        "winner_team", "number_of_games", "game1_team1", "game1_team2",
        "game2_team1", "game2_team2", "game3_team1", "game3_team2", "score",
    }
    historical_key_set = {
        "team1_player1_id", "team1_player2_id",
        "team2_player1_id", "team2_player2_id",
    }
    direct_identifier_set = {
        "match_id", "tournament_id", "team1", "team2",
        "team1_player1", "team1_player2",
        "team2_player1", "team2_player2",
        "source_endpoint",
    }

    for column in bwf_raw.columns:
        if column == "winner":
            role = "Target"
            action = "Recode as Team 1 win = 1, Team 2 win = 0"
            model_allowed = "Target only"
        elif column in post_match_set:
            role = "Post-match audit variable"
            action = "Use for match-quality checks; exclude from predictors"
            model_allowed = "No"
        elif column in historical_key_set:
            role = "Historical feature-engineering key"
            action = "Retain while constructing history; exclude as direct predictor"
            model_allowed = "No direct use"
        elif column in direct_identifier_set:
            role = "Identifier / metadata"
            action = "Retain for audit or display; exclude as direct predictor"
            model_allowed = "No direct use"
        elif column == "start_datetime_utc":
            role = "Metadata"
            action = "Audit time availability; do not assume within-day order"
            model_allowed = "No"
        elif column == "date":
            role = "Chronological key"
            action = "Use for historical features and chronological splitting"
            model_allowed = "No direct raw date"
        else:
            role = "Candidate pre-match context"
            action = "Retain or derive a lower-cardinality context feature"
            model_allowed = "Yes"

        bwf_dictionary_rows.append({
            "variable": column,
            "description": bwf_descriptions.get(column, ""),
            "raw_dtype": str(bwf_raw[column].dtype),
            "unique_values": bwf_raw[column].nunique(dropna=True),
            "missing": bwf_raw[column].isna().sum(),
            "missing_pct": bwf_raw[column].isna().mean() * 100,
            "role": role,
            "preprocessing_action": action,
            "allowed_in_model": model_allowed,
        })

    bwf_data_dictionary = pd.DataFrame(bwf_dictionary_rows)
    display(bwf_data_dictionary)
    bwf_data_dictionary.to_csv(
        BWF_OUT / "bwf_variable_dictionary.csv", index=False
    )

    # %% [markdown]
    # ## 9. BWF structure, chronology and distributions

    # %%

    bwf = bwf_raw.copy()
    bwf["date"] = pd.to_datetime(bwf["date"], errors="coerce")
    bwf["start_datetime_utc"] = pd.to_datetime(
        bwf["start_datetime_utc"], errors="coerce", utc=True
    )

    bwf_overview = pd.DataFrame({
        "measure": [
            "Rows",
            "Columns",
            "Date start",
            "Date end",
            "Unique tournaments",
            "Unique match IDs",
            "Duplicate match IDs",
            "Exact duplicate rows",
        ],
        "value": [
            len(bwf),
            bwf.shape[1],
            bwf["date"].min().date(),
            bwf["date"].max().date(),
            bwf["tournament_id"].nunique(),
            bwf["match_id"].nunique(),
            bwf["match_id"].duplicated().sum(),
            bwf.duplicated().sum(),
        ],
    })
    display(bwf_overview)

    matches_by_year = bwf["date"].dt.year.value_counts().sort_index()
    matches_by_discipline = bwf["discipline"].value_counts()

    display(matches_by_year.to_frame("matches"))
    display(matches_by_discipline.to_frame("matches"))
    display(bwf["tier"].value_counts().to_frame("matches"))
    display(bwf["round"].value_counts(dropna=False).to_frame("matches"))

    plt.figure(figsize=(8, 4.5))
    matches_by_year.plot(kind="bar")
    plt.title("BWF matches by year")
    plt.xlabel("Year")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    plt.figure(figsize=(8, 4.5))
    matches_by_discipline.plot(kind="bar")
    plt.title("BWF matches by discipline")
    plt.xlabel("Discipline")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    # %% [markdown]
    # ## 10. BWF missingness and structural validation

    # %%

    display(column_profile(bwf))

    singles_mask = bwf["discipline"].isin(["MS", "WS"])
    doubles_mask = bwf["discipline"].isin(["MD", "WD", "XD"])

    expected_winner_team = np.where(
        bwf["winner"].eq(1),
        bwf["team1"],
        bwf["team2"],
    )

    structure_checks = pd.DataFrame({
        "check": [
            "Singles rows",
            "Doubles rows",
            "Singles with unexpected Team 1 Player 2",
            "Singles with unexpected Team 2 Player 2",
            "Doubles missing Team 1 Player 2",
            "Doubles missing Team 2 Player 2",
            "Doubles repeating the same player on Team 1",
            "Doubles repeating the same player on Team 2",
            "Player appearing on both sides",
            "winner_team inconsistent with winner",
            "Missing round",
        ],
        "count": [
            int(singles_mask.sum()),
            int(doubles_mask.sum()),
            int(bwf.loc[singles_mask, "team1_player2_id"].notna().sum()),
            int(bwf.loc[singles_mask, "team2_player2_id"].notna().sum()),
            int(bwf.loc[doubles_mask, "team1_player2_id"].isna().sum()),
            int(bwf.loc[doubles_mask, "team2_player2_id"].isna().sum()),
            int((doubles_mask & bwf["team1_player1_id"].eq(bwf["team1_player2_id"])).sum()),
            int((doubles_mask & bwf["team2_player1_id"].eq(bwf["team2_player2_id"])).sum()),
            int((
                bwf[["team1_player1_id", "team1_player2_id"]]
                .apply(lambda r: set(r.dropna().astype(int)), axis=1)
                .combine(
                    bwf[["team2_player1_id", "team2_player2_id"]]
                    .apply(lambda r: set(r.dropna().astype(int)), axis=1),
                    lambda a, b: len(a.intersection(b)) > 0,
                )
            ).sum()),
            int((bwf["winner_team"].astype(str).str.strip() != pd.Series(expected_winner_team).astype(str).str.strip()).sum()),
            int(bwf["round"].isna().sum()),
        ],
    })
    display(structure_checks)

    # %% [markdown]
    # ## 11. Winner balance and qualification records

    # %%

    winner_summary = bwf["winner"].value_counts().sort_index().to_frame("matches")
    winner_summary["percentage"] = winner_summary["matches"].div(len(bwf)).mul(100)
    winner_summary.index = ["Team 1 wins", "Team 2 wins"]
    display(winner_summary)

    display(pd.crosstab(bwf["discipline"], bwf["winner"], normalize="index").rename(
        columns={1: "Team 1 win rate", 2: "Team 2 win rate"}
    ))

    display(bwf["is_qualification"].value_counts().rename(index={False: "Main draw", True: "Qualification"}).to_frame("matches"))

    # %% [markdown]
    # ## BWF report-ready visualisations

    # %%

    discipline_tier = pd.crosstab(bwf["discipline"], bwf["tier"])
    discipline_tier.plot(kind="bar", figsize=(11, 5))
    plt.title("BWF matches by discipline and tournament tier")
    plt.xlabel("Discipline")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    plt.legend(title="Tier", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    games_per_match = bwf["number_of_games"].value_counts().sort_index()
    plt.figure(figsize=(8, 4.5))
    games_per_match.plot(kind="bar")
    plt.title("BWF games per recorded match")
    plt.xlabel("Number of games")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    plt.figure(figsize=(8, 4.5))
    bwf["round"].value_counts(dropna=False).plot(kind="bar")
    plt.title("BWF match records by round")
    plt.xlabel("Round")
    plt.ylabel("Matches")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    plt.figure(figsize=(7, 4.5))
    bwf["winner"].map({1: "Team 1", 2: "Team 2"}).value_counts().plot(kind="bar")
    plt.title("BWF target balance")
    plt.xlabel("Winning side")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    # %%

    winning_country_occurrences = []

    for row in bwf.itertuples(index=False):
        if row.winner == 1:
            countries = [
                row.team1_player1_country,
                row.team1_player2_country,
            ]
        else:
            countries = [
                row.team2_player1_country,
                row.team2_player2_country,
            ]

        winning_country_occurrences.extend([
            country
            for country in countries
            if pd.notna(country) and str(country).strip()
        ])

    winning_country_counts = pd.Series(
        winning_country_occurrences,
        name="country",
    ).value_counts().head(15)

    display(winning_country_counts.to_frame("winning_player_occurrences"))

    plt.figure(figsize=(9, 5))
    winning_country_counts.sort_values().plot(kind="barh")
    plt.title("Top winning-player nationality occurrences")
    plt.xlabel("Winning-player occurrences")
    plt.ylabel("Country code")
    plt.tight_layout()
    plt.show() if show_outputs else plt.close()

    print(
        "This chart counts the countries represented by players on the official winning side. "
        "It is descriptive and does not imply nationality causes performance."
    )

    # %% [markdown]
    # ## Cross-year identity and tournament consistency checks

    # %%

    player_records = []

    player_column_groups = [
        ("team1_player1_id", "team1_player1", "team1_player1_country"),
        ("team1_player2_id", "team1_player2", "team1_player2_country"),
        ("team2_player1_id", "team2_player1", "team2_player1_country"),
        ("team2_player2_id", "team2_player2", "team2_player2_country"),
    ]

    for id_column, name_column, country_column in player_column_groups:
        temp = bwf[["date", id_column, name_column, country_column]].copy()
        temp.columns = ["date", "player_id", "player_name", "country"]
        temp = temp.dropna(subset=["player_id"])
        temp["player_id"] = temp["player_id"].astype("int64")
        player_records.append(temp)

    player_identity = pd.concat(player_records, ignore_index=True)

    player_id_name_audit = (
        player_identity.groupby("player_id")["player_name"]
        .nunique(dropna=True)
        .sort_values(ascending=False)
        .to_frame("unique_names")
    )
    player_id_name_anomalies = player_id_name_audit[
        player_id_name_audit["unique_names"] > 1
    ]

    player_name_id_audit = (
        player_identity.groupby("player_name")["player_id"]
        .nunique(dropna=True)
        .sort_values(ascending=False)
        .to_frame("unique_ids")
    )
    player_name_id_anomalies = player_name_id_audit[
        player_name_id_audit["unique_ids"] > 1
    ]

    player_country_audit = (
        player_identity.groupby("player_id")["country"]
        .nunique(dropna=True)
        .sort_values(ascending=False)
        .to_frame("unique_country_codes")
    )
    player_country_changes = player_country_audit[
        player_country_audit["unique_country_codes"] > 1
    ]

    tournament_identity_audit = (
        bwf.groupby("tournament_id")
        .agg(
            unique_names=("tournament", "nunique"),
            unique_tiers=("tier", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
    )
    tournament_identity_anomalies = tournament_identity_audit[
        (tournament_identity_audit["unique_names"] > 1)
        | (tournament_identity_audit["unique_tiers"] > 1)
    ]

    print("Player IDs mapped to multiple names:", len(player_id_name_anomalies))
    display(player_id_name_anomalies.head(20))

    print("Player names mapped to multiple IDs:", len(player_name_id_anomalies))
    display(player_name_id_anomalies.head(20))

    print("Player IDs with multiple country codes:", len(player_country_changes))
    display(player_country_changes.head(20))

    print("Tournament IDs with inconsistent names or tiers:", len(tournament_identity_anomalies))
    display(tournament_identity_anomalies.head(20))

    player_id_name_anomalies.to_csv(
        BWF_OUT / "bwf_player_id_name_anomalies.csv"
    )
    player_name_id_anomalies.to_csv(
        BWF_OUT / "bwf_player_name_id_anomalies.csv"
    )
    player_country_changes.to_csv(
        BWF_OUT / "bwf_player_country_changes.csv"
    )
    tournament_identity_anomalies.to_csv(
        BWF_OUT / "bwf_tournament_identity_anomalies.csv"
    )

    print(
        "Player ID remains the identity key. Shared names are not automatically merged, "
        "and country changes are retained as audit findings rather than treated as errors."
    )

    # %% [markdown]
    # ## 12. Actual match-time availability

    # %%

    midnight_time = pd.Timestamp("00:00:00").time()
    midnight_count = bwf["start_datetime_utc"].dt.time.eq(midnight_time).sum()

    time_summary = pd.DataFrame({
        "measure": [
            "Records",
            "Midnight timestamps",
            "Non-midnight timestamps",
            "Midnight percentage",
        ],
        "value": [
            len(bwf),
            midnight_count,
            len(bwf) - midnight_count,
            midnight_count / len(bwf) * 100,
        ],
    })
    display(time_summary)

    player_date_rows = []
    for side in ["team1", "team2"]:
        for player_number in [1, 2]:
            column = f"{side}_player{player_number}_id"
            temp = bwf[["date", column]].dropna().rename(columns={column: "player_id"})
            player_date_rows.append(temp)

    player_dates = pd.concat(player_date_rows, ignore_index=True)
    multiple_same_date = (
        player_dates.groupby(["player_id", "date"]).size().gt(1).sum()
    )

    print("Player-date combinations with more than one recorded match:", multiple_same_date)
    print(
        "Decision: exact within-day order will not be assumed. "
        "All matches on one date will use historical state from strictly earlier dates."
    )

    # %% [markdown]
    # ## 13. Identify standard completed matches

    # %%

    def is_valid_completed_badminton_game(score_a, score_b):
        if pd.isna(score_a) or pd.isna(score_b):
            return False

        high = max(int(score_a), int(score_b))
        low = min(int(score_a), int(score_b))

        if high == 21:
            return low <= 19
        if 22 <= high <= 29:
            return high - low == 2
        if high == 30:
            return low in {28, 29}
        return False

    def is_standard_completed_match(row):
        try:
            number_of_games = int(row["number_of_games"])
        except (TypeError, ValueError):
            return False

        if number_of_games not in {2, 3}:
            return False

        team1_game_wins = 0
        team2_game_wins = 0

        for game_number in range(1, number_of_games + 1):
            team1_score = row[f"game{game_number}_team1"]
            team2_score = row[f"game{game_number}_team2"]

            if not is_valid_completed_badminton_game(team1_score, team2_score):
                return False

            team1_game_wins += int(team1_score > team2_score)
            team2_game_wins += int(team2_score > team1_score)

        official_winner_game_wins = (
            team1_game_wins if int(row["winner"]) == 1 else team2_game_wins
        )

        return (
            max(team1_game_wins, team2_game_wins) == 2
            and official_winner_game_wins == 2
        )

    bwf["standard_completed_match"] = bwf.apply(is_standard_completed_match, axis=1)

    completion_summary = bwf["standard_completed_match"].value_counts().rename(
        index={True: "Standard completed", False: "Potential retirement/incomplete"}
    ).to_frame("matches")
    completion_summary["percentage"] = completion_summary["matches"].div(len(bwf)).mul(100)

    display(completion_summary)
    display(
        bwf.loc[
            ~bwf["standard_completed_match"],
            [
                "match_id", "date", "discipline", "winner", "number_of_games",
                "game1_team1", "game1_team2", "game2_team1", "game2_team2",
                "game3_team1", "game3_team2", "score",
            ],
        ].head(15)
    )

    print(
        "The flag is conservative. Suspicious records are preserved in the validated dataset "
        "but excluded from the primary model-ready table."
    )

    # %% [markdown]
    # ## BWF match-quality breakdown

    # %%

    bwf["year"] = bwf["date"].dt.year

    quality_by_year = (
        bwf.groupby("year")["standard_completed_match"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "records",
            "sum": "standard_completed",
            "mean": "standard_completed_rate",
        })
    )
    quality_by_year["potential_incomplete"] = (
        quality_by_year["records"] - quality_by_year["standard_completed"]
    )

    quality_by_discipline = (
        bwf.groupby("discipline")["standard_completed_match"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "records",
            "sum": "standard_completed",
            "mean": "standard_completed_rate",
        })
    )
    quality_by_discipline["potential_incomplete"] = (
        quality_by_discipline["records"]
        - quality_by_discipline["standard_completed"]
    )

    quality_by_tier = (
        bwf.groupby("tier")["standard_completed_match"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "records",
            "sum": "standard_completed",
            "mean": "standard_completed_rate",
        })
    )
    quality_by_tier["potential_incomplete"] = (
        quality_by_tier["records"] - quality_by_tier["standard_completed"]
    )

    display(quality_by_year)
    display(quality_by_discipline)
    display(quality_by_tier)

    quality_by_year.to_csv(BWF_OUT / "bwf_match_quality_by_year.csv")
    quality_by_discipline.to_csv(BWF_OUT / "bwf_match_quality_by_discipline.csv")
    quality_by_tier.to_csv(BWF_OUT / "bwf_match_quality_by_tier.csv")

    # %% [markdown]
    # ## 14. BWF deterministic cleaning and quality flags

    # %%

    bwf_clean = bwf.copy()

    for column in bwf_clean.select_dtypes(include="object").columns:
        bwf_clean[column] = bwf_clean[column].astype("string").str.strip()

    bwf_clean["round_missing_original"] = bwf_clean["round"].isna()
    bwf_clean["round"] = bwf_clean["round"].fillna("Unknown")

    country_columns = [
        "team1_player1_country", "team1_player2_country",
        "team2_player1_country", "team2_player2_country",
    ]
    for column in country_columns:
        bwf_clean[f"{column}_missing_original"] = bwf_clean[column].isna()
        bwf_clean[column] = bwf_clean[column].fillna("Unknown")

    tier_map = {
        "BWF Tour Super 100": "Super 100",
        "HSBC BWF World Tour Super 300": "Super 300",
        "HSBC BWF World Tour Super 500": "Super 500",
        "HSBC BWF World Tour Super 750": "Super 750",
        "HSBC BWF World Tour Super 1000": "Super 1000",
        "HSBC BWF World Tour Finals": "World Tour Finals",
    }
    bwf_clean["tier_category"] = bwf_clean["tier"].map(tier_map).fillna(bwf_clean["tier"])

    bwf_clean["host_country"] = (
        bwf_clean["host_location"]
        .astype("string")
        .str.split(",")
        .str[-1]
        .str.strip()
        .fillna("Unknown")
    )

    bwf_clean["year"] = bwf_clean["date"].dt.year.astype("int64")
    bwf_clean["team1_win"] = bwf_clean["winner"].eq(1).astype("int64")

    valid_player_structure = pd.Series(True, index=bwf_clean.index)
    valid_player_structure &= ~(
        doubles_mask & (
            bwf_clean["team1_player2_id"].isna()
            | bwf_clean["team2_player2_id"].isna()
        )
    )
    valid_player_structure &= ~(
        singles_mask & (
            bwf_clean["team1_player2_id"].notna()
            | bwf_clean["team2_player2_id"].notna()
        )
    )

    bwf_clean["valid_player_structure"] = valid_player_structure
    bwf_clean["model_eligible"] = (
        bwf_clean["standard_completed_match"]
        & bwf_clean["valid_player_structure"]
        & bwf_clean["date"].notna()
        & bwf_clean["winner"].isin([1, 2])
    )

    display(bwf_clean[["standard_completed_match", "valid_player_structure", "model_eligible"]].sum().to_frame("records"))

    # %%

    # These columns are retained for auditing but excluded from pre-match predictors.
    bwf_post_match_columns = [
        "winner",
        "winner_team",
        "number_of_games",
        "game1_team1",
        "game1_team2",
        "game2_team1",
        "game2_team2",
        "game3_team1",
        "game3_team2",
        "score",
    ]

    bwf_identifier_columns = [
        "match_id",
        "tournament_id",
        "team1_player1_id",
        "team1_player2_id",
        "team2_player1_id",
        "team2_player2_id",
        "team1",
        "team2",
        "team1_player1",
        "team1_player2",
        "team2_player1",
        "team2_player2",
        "source_endpoint",
    ]

    print("Post-match columns excluded from predictors:")
    print(bwf_post_match_columns)
    print("\nIdentifiers retained for audit/feature construction but excluded as direct predictors:")
    print(bwf_identifier_columns)

    # %% [markdown]
    # # Part C — Leakage-Safe BWF Pre-Match Feature Engineering

    # %% [markdown]
    # ## 15. Feature-engineering policy
    #
    # For every date:
    #
    # 1. Generate the features for **all matches on that date** using history from earlier dates only.
    # 2. Save those pre-match feature rows.
    # 3. Update ratings, form, experience, rest and head-to-head states after the complete date batch.
    # 4. Continue to the next date.
    #
    # This avoids inventing an unsupported within-day match order.

    # %%
    def row_team_players(row, side):
        player1 = int(getattr(row, f"{side}_player1_id"))
        player2 = getattr(row, f"{side}_player2_id")

        if pd.isna(player2):
            return (player1,)

        return tuple(sorted((player1, int(player2))))

    def history_win_rate(history, window=None):
        if not history:
            return np.nan

        values = list(history)
        if window is not None:
            values = values[-window:]

        return float(np.mean(values)) if values else np.nan

    def mean_without_missing(values):
        usable = [value for value in values if not pd.isna(value)]
        return float(np.mean(usable)) if usable else np.nan

    def build_date_batched_bwf_features(
        matches,
        default_elo=1500.0,
        k_factor=32.0,
        cold_start_threshold=5,
    ):
        player_elo = defaultdict(lambda: default_elo)
        partnership_elo = defaultdict(lambda: default_elo)

        player_history = defaultdict(lambda: deque(maxlen=10))
        partnership_history = defaultdict(lambda: deque(maxlen=10))

        player_matches = defaultdict(int)
        player_wins = defaultdict(int)
        partnership_matches = defaultdict(int)

        player_last_date = {}

        h2h_wins = defaultdict(lambda: defaultdict(int))
        h2h_matches = defaultdict(int)

        feature_rows = []
        sorted_matches = matches.sort_values(["date", "match_id"]).copy()

        for match_date, date_matches in sorted_matches.groupby("date", sort=True):
            player_elo_delta = defaultdict(float)
            partnership_elo_delta = defaultdict(float)
            date_updates = []

            # Features for all matches on this date use state from earlier dates only.
            for row in date_matches.sort_values("match_id").itertuples(index=False):
                discipline = row.discipline
                is_doubles = discipline in {"MD", "WD", "XD"}

                team1_players = row_team_players(row, "team1")
                team2_players = row_team_players(row, "team2")

                team1_player_keys = [(discipline, player) for player in team1_players]
                team2_player_keys = [(discipline, player) for player in team2_players]

                team1_key = (discipline, team1_players)
                team2_key = (discipline, team2_players)

                team1_individual_elo = np.mean([
                    player_elo[key] for key in team1_player_keys
                ])
                team2_individual_elo = np.mean([
                    player_elo[key] for key in team2_player_keys
                ])

                if is_doubles:
                    team1_partnership_elo = partnership_elo[team1_key]
                    team2_partnership_elo = partnership_elo[team2_key]
                    partnership_elo_diff = (
                        team1_partnership_elo - team2_partnership_elo
                    )
                    team1_partnership_form_5 = history_win_rate(
                        partnership_history[team1_key], 5
                    )
                    team2_partnership_form_5 = history_win_rate(
                        partnership_history[team2_key], 5
                    )
                    partnership_form_5_diff = safe_difference(
                        team1_partnership_form_5,
                        team2_partnership_form_5,
                    )
                    team1_partnership_match_count = partnership_matches[team1_key]
                    team2_partnership_match_count = partnership_matches[team2_key]
                    partnership_matches_diff = (
                        team1_partnership_match_count
                        - team2_partnership_match_count
                    )
                else:
                    # Partnership variables are not applicable to singles.
                    team1_partnership_elo = np.nan
                    team2_partnership_elo = np.nan
                    partnership_elo_diff = 0.0
                    team1_partnership_form_5 = np.nan
                    team2_partnership_form_5 = np.nan
                    partnership_form_5_diff = 0.0
                    team1_partnership_match_count = 0
                    team2_partnership_match_count = 0
                    partnership_matches_diff = 0

                team1_form_5 = mean_without_missing([
                    history_win_rate(player_history[key], 5)
                    for key in team1_player_keys
                ])
                team2_form_5 = mean_without_missing([
                    history_win_rate(player_history[key], 5)
                    for key in team2_player_keys
                ])

                team1_form_10 = mean_without_missing([
                    history_win_rate(player_history[key], 10)
                    for key in team1_player_keys
                ])
                team2_form_10 = mean_without_missing([
                    history_win_rate(player_history[key], 10)
                    for key in team2_player_keys
                ])

                team1_experience = float(np.mean([
                    player_matches[key] for key in team1_player_keys
                ]))
                team2_experience = float(np.mean([
                    player_matches[key] for key in team2_player_keys
                ]))

                team1_career_win_rate = mean_without_missing([
                    player_wins[key] / player_matches[key]
                    if player_matches[key] > 0 else np.nan
                    for key in team1_player_keys
                ])
                team2_career_win_rate = mean_without_missing([
                    player_wins[key] / player_matches[key]
                    if player_matches[key] > 0 else np.nan
                    for key in team2_player_keys
                ])

                team1_rest_values = [
                    (match_date - player_last_date[key]).days
                    if key in player_last_date else np.nan
                    for key in team1_player_keys
                ]
                team2_rest_values = [
                    (match_date - player_last_date[key]).days
                    if key in player_last_date else np.nan
                    for key in team2_player_keys
                ]

                team1_rest_mean = mean_without_missing(team1_rest_values)
                team2_rest_mean = mean_without_missing(team2_rest_values)

                valid_team1_rest = [
                    value for value in team1_rest_values if not pd.isna(value)
                ]
                valid_team2_rest = [
                    value for value in team2_rest_values if not pd.isna(value)
                ]

                team1_rest_min = (
                    min(valid_team1_rest) if valid_team1_rest else np.nan
                )
                team2_rest_min = (
                    min(valid_team2_rest) if valid_team2_rest else np.nan
                )

                h2h_pair = tuple(sorted((team1_players, team2_players)))
                h2h_key = (discipline, h2h_pair)

                team1_h2h_wins = h2h_wins[h2h_key][team1_players]
                team2_h2h_wins = h2h_wins[h2h_key][team2_players]

                team1_win = int(row.winner == 1)

                feature_rows.append({
                    "match_id": row.match_id,
                    "date": match_date,
                    "year": match_date.year,
                    "discipline": discipline,
                    "is_doubles": int(is_doubles),
                    "tier": row.tier,
                    "tier_category": row.tier_category,
                    "round": row.round,
                    "is_qualification": bool(row.is_qualification),
                    "host_location": row.host_location,
                    "host_country": row.host_country,
                    "tournament_id": row.tournament_id,
                    "tournament": row.tournament,
                    "team1": row.team1,
                    "team2": row.team2,

                    "team1_individual_elo": team1_individual_elo,
                    "team2_individual_elo": team2_individual_elo,
                    "individual_elo_diff": (
                        team1_individual_elo - team2_individual_elo
                    ),

                    "team1_partnership_elo": team1_partnership_elo,
                    "team2_partnership_elo": team2_partnership_elo,
                    "partnership_elo_diff": partnership_elo_diff,

                    "team1_recent_form_5": team1_form_5,
                    "team2_recent_form_5": team2_form_5,
                    "recent_form_5_diff": safe_difference(
                        team1_form_5, team2_form_5
                    ),

                    "team1_recent_form_10": team1_form_10,
                    "team2_recent_form_10": team2_form_10,
                    "recent_form_10_diff": safe_difference(
                        team1_form_10, team2_form_10
                    ),

                    "team1_partnership_form_5": team1_partnership_form_5,
                    "team2_partnership_form_5": team2_partnership_form_5,
                    "partnership_form_5_diff": partnership_form_5_diff,

                    "team1_player_experience_mean": team1_experience,
                    "team2_player_experience_mean": team2_experience,
                    "player_experience_diff": (
                        team1_experience - team2_experience
                    ),

                    "team1_career_win_rate_mean": team1_career_win_rate,
                    "team2_career_win_rate_mean": team2_career_win_rate,
                    "career_win_rate_diff": safe_difference(
                        team1_career_win_rate,
                        team2_career_win_rate,
                    ),

                    "team1_partnership_matches": team1_partnership_match_count,
                    "team2_partnership_matches": team2_partnership_match_count,
                    "partnership_matches_diff": partnership_matches_diff,

                    "team1_rest_days_mean": team1_rest_mean,
                    "team2_rest_days_mean": team2_rest_mean,
                    "rest_days_mean_diff": safe_difference(
                        team1_rest_mean, team2_rest_mean
                    ),

                    "team1_rest_days_min": team1_rest_min,
                    "team2_rest_days_min": team2_rest_min,
                    "rest_days_min_diff": safe_difference(
                        team1_rest_min, team2_rest_min
                    ),

                    "h2h_matches": h2h_matches[h2h_key],
                    "team1_h2h_wins": team1_h2h_wins,
                    "team2_h2h_wins": team2_h2h_wins,
                    "h2h_win_diff": (
                        team1_h2h_wins - team2_h2h_wins
                    ),

                    "team1_cold_start": int(any(
                        player_matches[key] < cold_start_threshold
                        for key in team1_player_keys
                    )),
                    "team2_cold_start": int(any(
                        player_matches[key] < cold_start_threshold
                        for key in team2_player_keys
                    )),

                    "team1_win": team1_win,
                })

                expected_team1_player = 1 / (
                    1 + 10 ** (
                        (team2_individual_elo - team1_individual_elo) / 400
                    )
                )
                player_delta = (
                    k_factor * (team1_win - expected_team1_player)
                )

                for key in team1_player_keys:
                    player_elo_delta[key] += player_delta
                for key in team2_player_keys:
                    player_elo_delta[key] -= player_delta

                if is_doubles:
                    expected_team1_partnership = 1 / (
                        1 + 10 ** (
                            (
                                team2_partnership_elo
                                - team1_partnership_elo
                            ) / 400
                        )
                    )
                    partnership_delta = (
                        k_factor
                        * (team1_win - expected_team1_partnership)
                    )
                    partnership_elo_delta[team1_key] += partnership_delta
                    partnership_elo_delta[team2_key] -= partnership_delta

                date_updates.append({
                    "discipline": discipline,
                    "is_doubles": is_doubles,
                    "team1_players": team1_players,
                    "team2_players": team2_players,
                    "team1_player_keys": team1_player_keys,
                    "team2_player_keys": team2_player_keys,
                    "team1_key": team1_key,
                    "team2_key": team2_key,
                    "team1_win": team1_win,
                    "h2h_key": h2h_key,
                    "date": match_date,
                })

            for key, delta in player_elo_delta.items():
                player_elo[key] += delta

            for key, delta in partnership_elo_delta.items():
                partnership_elo[key] += delta

            for update in date_updates:
                team1_win = update["team1_win"]

                for key in update["team1_player_keys"]:
                    player_history[key].append(team1_win)
                    player_matches[key] += 1
                    player_wins[key] += team1_win
                    player_last_date[key] = update["date"]

                for key in update["team2_player_keys"]:
                    player_history[key].append(1 - team1_win)
                    player_matches[key] += 1
                    player_wins[key] += 1 - team1_win
                    player_last_date[key] = update["date"]

                if update["is_doubles"]:
                    partnership_history[
                        update["team1_key"]
                    ].append(team1_win)
                    partnership_history[
                        update["team2_key"]
                    ].append(1 - team1_win)
                    partnership_matches[update["team1_key"]] += 1
                    partnership_matches[update["team2_key"]] += 1

                h2h_wins[
                    update["h2h_key"]
                ][update["team1_players"]] += team1_win
                h2h_wins[
                    update["h2h_key"]
                ][update["team2_players"]] += 1 - team1_win
                h2h_matches[update["h2h_key"]] += 1

        return pd.DataFrame(feature_rows)

    # %% [markdown]
    # ## 16. Generate the BWF pre-match feature table

    # %%

    bwf_model_eligible = (
        bwf_clean.loc[bwf_clean["model_eligible"]]
        .sort_values(["date", "match_id"])
        .reset_index(drop=True)
    )

    bwf_features = build_date_batched_bwf_features(bwf_model_eligible)

    bwf_features["split"] = np.select(
        [
            bwf_features["year"].eq(2021),
            bwf_features["year"].isin([2022, 2023]),
            bwf_features["year"].eq(2024),
            bwf_features["year"].eq(2025),
        ],
        [
            "warmup",
            "train",
            "validation",
            "test",
        ],
        default="exclude",
    )

    display(bwf_features.head())
    display(bwf_features.groupby("split").agg(
        records=("match_id", "size"),
        tournaments=("tournament_id", "nunique"),
    ))
    display(pd.crosstab(
        bwf_features["split"],
        bwf_features["team1_win"],
        normalize="index",
    ).rename(columns={0: "Team 2 win rate", 1: "Team 1 win rate"}))

    print("Feature rows:", len(bwf_features))
    print("Warm-up rows:", (bwf_features["split"] == "warmup").sum())
    print("Rows used for model development/evaluation:", (bwf_features["split"] != "warmup").sum())

    # %% [markdown]
    # ## 17. Pre-match feature audit

    # %%

    feature_missingness = (
        bwf_features.isna().mean().mul(100).sort_values(ascending=False)
    )
    display(feature_missingness[feature_missingness > 0].to_frame("missing_pct"))

    cold_start_summary = pd.DataFrame({
        "measure": [
            "Team 1 cold-start rows",
            "Team 2 cold-start rows",
            "Either team cold-start",
            "Rows with prior H2H",
        ],
        "records": [
            bwf_features["team1_cold_start"].sum(),
            bwf_features["team2_cold_start"].sum(),
            (
                bwf_features["team1_cold_start"].eq(1)
                | bwf_features["team2_cold_start"].eq(1)
            ).sum(),
            bwf_features["h2h_matches"].gt(0).sum(),
        ],
    })
    display(cold_start_summary)

    print(
        "Missing engineered values are expected for cold-start players or partnerships. "
        "They must be imputed inside the training pipeline, not before the chronological split."
    )

    # %% [markdown]
    # ## BWF engineered-feature quality by year and discipline

    # %%

    bwf_features["either_team_cold_start"] = (
        bwf_features["team1_cold_start"].eq(1)
        | bwf_features["team2_cold_start"].eq(1)
    ).astype(int)

    cold_start_by_year = (
        bwf_features.groupby("year")
        .agg(
            records=("match_id", "size"),
            either_team_cold_start_rate=("either_team_cold_start", "mean"),
            prior_h2h_rate=("h2h_matches", lambda x: x.gt(0).mean()),
        )
    )

    cold_start_by_discipline = (
        bwf_features.groupby("discipline")
        .agg(
            records=("match_id", "size"),
            either_team_cold_start_rate=("either_team_cold_start", "mean"),
            prior_h2h_rate=("h2h_matches", lambda x: x.gt(0).mean()),
        )
    )

    display(cold_start_by_year)
    display(cold_start_by_discipline)

    cold_start_by_year.to_csv(
        BWF_OUT / "bwf_feature_quality_by_year.csv"
    )
    cold_start_by_discipline.to_csv(
        BWF_OUT / "bwf_feature_quality_by_discipline.csv"
    )

    # %% [markdown]
    # ## 18. BWF model feature manifest and preprocessing template

    # %%
    bwf_target_column = "team1_win"

    bwf_categorical_features = [
        "discipline",
        "tier_category",
        "round",
        "is_qualification",
        "host_country",
    ]

    bwf_numeric_features = [
        "is_doubles",
        "individual_elo_diff",
        "partnership_elo_diff",
        "recent_form_5_diff",
        "recent_form_10_diff",
        "partnership_form_5_diff",
        "player_experience_diff",
        "career_win_rate_diff",
        "partnership_matches_diff",
        "rest_days_mean_diff",
        "rest_days_min_diff",
        "h2h_matches",
        "h2h_win_diff",
        "team1_cold_start",
        "team2_cold_start",
    ]

    bwf_feature_manifest = {
        "target": bwf_target_column,
        "target_definition": "1 = Team 1 wins; 0 = Team 2 wins",
        "numeric_features": bwf_numeric_features,
        "categorical_features": bwf_categorical_features,
        "warmup_period": "2021",
        "training_period": "2022–2023",
        "validation_period": "2024",
        "final_test_period": "2025",
        "same_date_policy": (
            "All same-date rows use history from strictly earlier dates. "
            "State is updated only after the complete date batch."
        ),
        "singles_partnership_policy": (
            "Partnership features are not applicable to MS and WS. "
            "Their difference values are set to zero, and is_doubles identifies applicability."
        ),
        "excluded_post_match_columns": bwf_post_match_columns,
        "excluded_direct_identifiers": bwf_identifier_columns,
        "primary_match_filter": "standard_completed_match == True",
        "statistical_preprocessing_policy": (
            "Fit imputation, encoding and scaling inside each chronological training fold."
        ),
    }

    with open(
        BWF_OUT / "bwf_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(bwf_feature_manifest, file, indent=2)

    display(pd.DataFrame({
        "numeric_features": pd.Series(bwf_numeric_features),
        "categorical_features": pd.Series(bwf_categorical_features),
    }))

    def build_bwf_preprocessor():
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("numeric", numeric_pipeline, bwf_numeric_features),
            ("categorical", categorical_pipeline, bwf_categorical_features),
        ])

    bwf_preprocessor = build_bwf_preprocessor()
    print(bwf_preprocessor)

    print(
        "\nImportant: fit this preprocessor inside each chronological training fold. "
        "Do not fit it on 2024 or 2025 before final evaluation."
    )

    # %% [markdown]
    # ## 19. Export BWF validated, cleaned and feature-engineered files

    # %%

    bwf_clean.to_csv(BWF_OUT / "bwf_validated_with_flags.csv", index=False)
    bwf_model_eligible.to_csv(BWF_OUT / "bwf_model_eligible_clean.csv", index=False)
    bwf_features.to_csv(BWF_OUT / "bwf_prematch_features_2021_2025.csv", index=False)
    bwf_features.loc[bwf_features["split"] != "warmup"].to_csv(
        BWF_OUT / "bwf_model_rows_2022_2025.csv",
        index=False,
    )

    print("BWF outputs saved to:", BWF_OUT)

    # %% [markdown]
    # # Part D — Final Data-Quality Summary

    # %%

    quality_summary = pd.DataFrame([
        {
            "dataset": "Injury raw",
            "records": len(injury_raw),
            "unique_groups": injury_raw["athlete_id"].nunique(),
            "duplicate_rows": injury_raw.duplicated().sum(),
            "excluded_or_reserved": 0,
            "purpose": "Preserved source data",
        },
        {
            "dataset": "Injury development",
            "records": (injury_clean["split"] == "development").sum(),
            "unique_groups": injury_clean.loc[
                injury_clean["split"] == "development", "athlete_id"
            ].nunique(),
            "duplicate_rows": 0,
            "excluded_or_reserved": 0,
            "purpose": "Grouped CV and model development",
        },
        {
            "dataset": "Injury final test",
            "records": (injury_clean["split"] == "test").sum(),
            "unique_groups": injury_clean.loc[
                injury_clean["split"] == "test", "athlete_id"
            ].nunique(),
            "duplicate_rows": 0,
            "excluded_or_reserved": 0,
            "purpose": "Final athlete-disjoint evaluation",
        },
        {
            "dataset": "BWF raw",
            "records": len(bwf_raw),
            "unique_groups": bwf_raw["tournament_id"].nunique(),
            "duplicate_rows": bwf_raw.duplicated().sum(),
            "excluded_or_reserved": 0,
            "purpose": "Preserved source data",
        },
        {
            "dataset": "BWF model eligible",
            "records": int(bwf_clean["model_eligible"].sum()),
            "unique_groups": bwf_clean.loc[
                bwf_clean["model_eligible"], "tournament_id"
            ].nunique(),
            "duplicate_rows": 0,
            "excluded_or_reserved": int((~bwf_clean["model_eligible"]).sum()),
            "purpose": "Standard completed matches",
        },
        {
            "dataset": "BWF model rows 2022–2025",
            "records": int((bwf_features["split"] != "warmup").sum()),
            "unique_groups": bwf_features.loc[
                bwf_features["split"] != "warmup", "tournament_id"
            ].nunique(),
            "duplicate_rows": 0,
            "excluded_or_reserved": int((bwf_features["split"] == "warmup").sum()),
            "purpose": "Train, validation and final test",
        },
    ])

    display(quality_summary)
    quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

    # %% [markdown]
    # ## 20. Report-ready methodology summary

    # %%
    report_summary = f'''
    BWF MATCH DATA
    - Raw records: {len(bwf_raw):,}
    - Date range: {bwf["date"].min().date()} to {bwf["date"].max().date()}
    - Model-eligible standard completed matches:
      {int(bwf_clean["model_eligible"].sum()):,}
    - Potential retirement or incomplete records preserved but excluded:
      {int((~bwf_clean["standard_completed_match"]).sum()):,}
    - Historical warm-up: 2021
    - Training: 2022–2023
    - Validation: 2024
    - Final untouched test: 2025
    - Exact within-day match order was unavailable. Every same-date match uses
      historical state from strictly earlier dates, and state is updated after
      the complete date batch.
    - Game scores, number of games and score margins are audit variables only;
      they are not pre-match predictors.
    - Player IDs are used to construct history but are not direct predictors.
    - Partnership features are applied only to doubles disciplines.

    INJURY DATA
    - Raw and cleaned records: {len(injury_clean):,}
    - Athletes: {injury_clean["athlete_id"].nunique():,}
    - Development athletes: {len(development_athletes):,}
    - Final-test athletes: {len(test_athletes):,}
    - Selected athlete-group split seed: {selected_split_seed}
    - Athlete overlap between development and test: 0
    - Athlete and session identifiers are excluded as predictors.
    - Missing values remain in the cleaned file and will be imputed inside the
      grouped model-training pipeline.
    - Statistical outliers are reported and retained unless later validation
      supports a fold-contained treatment.

    LEAKAGE CONTROL
    - Post-match information is excluded from BWF predictors.
    - BWF historical features use strictly earlier dates.
    - Injury athletes are separated between development and final test.
    - Imputation, scaling, encoding, optional outlier treatment, resampling and
      tuning must be fitted inside training folds only.
    '''

    print(report_summary)

    # %% [markdown]
    # # Final notes before modelling
    #
    # ## BWF stream
    #
    # - Use `bwf_model_rows_2022_2025.csv` for modelling.
    # - Use 2022–2023 as training, 2024 as validation and 2025 as the final test.
    # - Do not use `winner`, `winner_team`, scores or game counts as predictors.
    # - Keep the preprocessing transformer inside each chronological model pipeline.
    # - Compare overall performance with discipline-level performance.
    #
    # ## Injury stream
    #
    # - Use `injury_development.csv` for grouped cross-validation and tuning.
    # - Use `injury_final_test.csv` only after selecting the final pipeline.
    # - Keep `athlete_id` as the grouping key but exclude it from predictors.
    # - Keep `session_id` for audit only and exclude it from predictors.
    # - Compare class weighting, sample weighting and SMOTENC during modelling.
    # - Apply resampling inside training folds only.
    #
    # This notebook intentionally stops before model training.

    # %% [markdown]
    # # Report-writing guidance
    #
    # The report should distinguish three data layers:
    #
    # 1. **Raw data** — preserved exactly as collected.
    # 2. **Validated and cleaned data** — deterministic formatting, audit flags and documented exclusions.
    # 3. **Model-ready data** — leakage-safe historical BWF features and fixed development/test assignments.
    #
    # ## Claims that are supported
    #
    # - The BWF data contains five disciplines across 2021–2025.
    # - The primary BWF modelling table excludes suspicious incomplete matches.
    # - The BWF target is close to balanced.
    # - The injury target is imbalanced and requires class-sensitive evaluation.
    # - Athlete-disjoint validation prevents the same athlete from appearing in development and final test data.
    # - Exact within-day BWF order is unavailable, so date-batched history is the conservative leakage-safe choice.
    #
    # ## Claims to avoid
    #
    # - Do not claim that score margins are pre-match predictors.
    # - Do not claim that SHAP or correlation proves causation.
    # - Do not describe the multi-sport injury data as clinically validated badminton evidence.
    # - Do not describe statistical outliers as errors unless source validation confirms this.
    # - Do not state that preprocessing produced zero missing values before the modelling pipeline; missing values are intentionally preserved for train-fold-only imputation.

    workflow_result = {
        "dataset_overview": dataset_overview,
        "quality_summary": quality_summary,
        "report_summary": report_summary,
        "injury": {
            "raw": injury_raw,
            "profile": injury_profile,
            "key_checks": injury_key_checks,
            "missing": injury_missing,
            "target_summary": injury_target_summary,
            "numeric_summary": injury_numeric_summary,
            "target_correlations": spearman_with_target,
            "outlier_report": injury_outlier_table,
            "range_report": range_report,
            "high_correlation_pairs": high_correlation_pairs,
            "cleaned": injury_clean,
            "development": injury_clean.loc[injury_clean["split"] == "development"].copy(),
            "final_test": injury_clean.loc[injury_clean["split"] == "test"].copy(),
            "split_summary": split_summary,
            "split_class_distribution": split_class_distribution,
            "split_sport_distribution": split_sport_distribution,
            "split_gender_distribution": split_gender_distribution,
            "candidate_split_report": candidate_split_report,
            "selected_split_seed": selected_split_seed,
            "manifest": injury_feature_manifest,
            "preprocessor": injury_preprocessor,
            "data_dictionary": injury_data_dictionary,
        },
        "bwf": {
            "raw": bwf_raw,
            "parsed": bwf,
            "overview": bwf_overview,
            "profile": column_profile(bwf),
            "structure_checks": structure_checks,
            "winner_summary": winner_summary,
            "completion_summary": completion_summary,
            "quality_by_year": quality_by_year,
            "quality_by_discipline": quality_by_discipline,
            "quality_by_tier": quality_by_tier,
            "identity_audits": {
                "player_id_name_anomalies": player_id_name_anomalies,
                "player_name_id_anomalies": player_name_id_anomalies,
                "player_country_changes": player_country_changes,
                "tournament_identity_anomalies": tournament_identity_anomalies,
            },
            "cleaned": bwf_clean,
            "model_eligible": bwf_model_eligible,
            "features": bwf_features,
            "model_rows": bwf_features.loc[bwf_features["split"] != "warmup"].copy(),
            "split_summary": bwf_features.groupby("split").agg(records=("match_id", "size"), tournaments=("tournament_id", "nunique")),
            "feature_missingness": feature_missingness,
            "cold_start_summary": cold_start_summary,
            "feature_quality_by_year": cold_start_by_year,
            "feature_quality_by_discipline": cold_start_by_discipline,
            "manifest": bwf_feature_manifest,
            "preprocessor": bwf_preprocessor,
            "data_dictionary": bwf_data_dictionary,
        },
    }
    return workflow_result
