from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.serving.errors import ArtifactUnavailable, ServiceFault


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INJURY_CLASS_NAMES = ("Healthy", "Low Risk", "High Risk/Injured")

INJURY_BINARY_CLASS_NAMES = ("Low Injury Risk", "High Injury Risk")

INJURY_BINARY_BUNDLE = (
    PROJECT_ROOT / "models" / "research" / "injury_binary_high_risk_bundle.joblib"
)


def _load(path: Path, description: str) -> Any:
    """Read one joblib artefact, distinguishing "absent" from "unreadable"."""
    if not path.exists():
        raise ArtifactUnavailable(f"{description} is missing at {path}.")
    try:
        return joblib.load(path)
    except Exception as error:  # a truncated, wrong-version or corrupt file
        raise ServiceFault(
            f"{description} at {path} could not be loaded: {error}"
        ) from error


@lru_cache(maxsize=1)
def load_bwf_bundle() -> dict[str, Any]:
    refit = PROJECT_ROOT / "models" / "baseline" / "bwf_baseline_bundle.joblib"
    path = refit if refit.exists() else PROJECT_ROOT / "models" / "bwf_champion_bundle.joblib"
    bundle = _load(path, "The BWF model bundle")
    _validate_bundle(bundle, expected_stream="bwf")
    return bundle


@lru_cache(maxsize=1)
def load_injury_bundle() -> dict[str, Any]:
    refit = PROJECT_ROOT / "models" / "champion" / "injury_champion_bundle.joblib"
    path = refit if refit.exists() else PROJECT_ROOT / "models" / "injury_champion_bundle.joblib"
    bundle = _load(path, "The injury model bundle")
    _validate_bundle(bundle, expected_stream="injury")
    return bundle


@lru_cache(maxsize=1)
def load_injury_binary_bundle() -> dict[str, Any]:
    """The binary Low/High Injury Risk research bundle.

    Served on separate ``/injury/binary/*`` routes while it is validated. It
    does not shadow, replace or fall back to the three-class champion: if it is
    absent the binary routes report it unavailable and the three-class stream
    carries on untouched.
    """
    bundle = _load(INJURY_BINARY_BUNDLE, "The binary injury model bundle")
    _validate_bundle(bundle, expected_stream="injury_binary")
    return bundle


def _validate_state(state: dict[str, Any]) -> None:
    """Shared by the loader and the reload check, so they cannot diverge.

    A reload that validated less strictly than the loader would admit a snapshot
    the next request then rejects, which is the failure the reload exists to
    prevent.
    """
    required = {
        "schema_version",
        "snapshot_as_of",
        "player_elo",
        "partnership_elo",
        "player_history",
        "partnership_history",
        "player_matches",
        "player_wins",
        "partnership_matches",
        "player_last_date",
        "h2h_wins",
        "h2h_matches",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ServiceFault(f"BWF state snapshot is missing keys: {missing}")


@lru_cache(maxsize=1)
def load_bwf_state() -> dict[str, Any]:
    path = PROJECT_ROOT / "artifacts" / "bwf_state_snapshot.joblib"
    state = _load(path, "The BWF rating state snapshot")
    _validate_state(state)
    return state


@lru_cache(maxsize=1)
def load_bwf_champion_bundle() -> dict[str, Any]:
    """The promoted champion: symmetric Glicko logistic regression.

    Loaded from ``models/champion/`` and never refitted. The earlier XGBoost
    baseline stays loadable alongside it
    through :func:`load_bwf_bundle`, because the project reports the baseline as its
    preserved baseline and a baseline you cannot load is not auditable.
    """
    path = PROJECT_ROOT / "models" / "champion" / "bwf_champion_bundle.joblib"
    bundle = _load(path, "The BWF champion model bundle")
    for required in ("model", "feature_columns", "model_version"):
        if required not in bundle:
            raise RuntimeError(f"The BWF champion bundle is missing '{required}'.")
    if bundle.get("stream") != "bwf":
        raise RuntimeError("The BWF champion bundle does not declare the bwf stream.")
    return bundle


@lru_cache(maxsize=1)
def load_bwf_glicko_state() -> dict[str, Any]:
    """The Glicko-2 rating state the champion features are constructed from."""
    path = PROJECT_ROOT / "artifacts" / "bwf_glicko_state_snapshot.joblib"
    state = _load(path, "The BWF Glicko-2 state snapshot")
    for required in ("players", "snapshot_as_of", "feature_code_version"):
        if required not in state:
            raise RuntimeError(f"The Glicko state snapshot is missing '{required}'.")
    return state


def _serving_model(bundle: dict[str, Any]) -> Any:
    return bundle.get("serving_model", bundle.get("calibrated_model"))


def _check_class_order(bundle: dict[str, Any], expected_stream: str) -> None:
    """Fail at load time if the bundle's class names cannot be trusted.

    ``class_names[i]`` is used everywhere as the name of ``predict_proba``'s
    column *i*. That holds only while the model's own ``classes_`` are the
    integer codes 0..n-1 in ascending order, which is what the training pipeline
    produced. A retrained bundle emitting string classes, a different order, or
    a different count breaks the assumption without breaking anything visibly,
    so it is asserted here rather than discovered from a mislabelled screening.
    """
    names = list(bundle["class_names"])
    if expected_stream == "injury_binary" and tuple(names) != INJURY_BINARY_CLASS_NAMES:
        raise ServiceFault(
            "The binary injury bundle's class_names are "
            f"{names!r}, but this service serves {list(INJURY_BINARY_CLASS_NAMES)!r}. "
            "Index 1 must be the High Injury Risk positive class; relabelling "
            "positionally would invert the screen, so the bundle is refused."
        )
    if expected_stream == "injury" and tuple(names) != INJURY_CLASS_NAMES:
        raise ServiceFault(
            "The injury bundle's class_names are "
            f"{names!r}, but this service serves {list(INJURY_CLASS_NAMES)!r}. "
            "Relabelling them positionally would attach every probability, "
            "conformal set and policy result to the wrong class, so the bundle "
            "is refused instead. Re-export it with matching names."
        )

    model = _serving_model(bundle)
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ServiceFault(
            f"The {expected_stream} serving model exposes no classes_, so the "
            "order of class_names cannot be verified against it."
        )

    observed = list(classes)
    if len(observed) != len(names):
        raise ServiceFault(
            f"The {expected_stream} bundle names {len(names)} classes but the "
            f"serving model has {len(observed)}: {observed!r}."
        )

    if all(isinstance(value, str) for value in observed):
        # A model that carries the labels itself is authoritative about order.
        if observed != names:
            raise ServiceFault(
                f"The {expected_stream} bundle's class_names {names!r} do not "
                f"match the serving model's own classes_ {observed!r}."
            )
        return

    # Otherwise the labels are integer codes and class_names is positional, so
    # the codes must be exactly 0..n-1 in ascending order for that to be true.
    try:
        codes = [int(value) for value in observed]
    except (TypeError, ValueError) as error:
        raise ServiceFault(
            f"The {expected_stream} serving model's classes_ {observed!r} are "
            "neither labels nor integer codes, so class_names cannot be mapped "
            "onto them."
        ) from error
    if codes != list(range(len(names))):
        raise ServiceFault(
            f"The {expected_stream} serving model's classes_ are {codes!r}. "
            "class_names is applied positionally, which is only correct for "
            f"{list(range(len(names)))!r}. Re-export the bundle with an "
            "explicit label order."
        )


def _validate_champion_bundle(bundle: dict[str, Any]) -> None:
    """The champion bundle carries a symmetric wrapper, not a plain sklearn pipeline.

    Validated separately from the baseline because the baseline checks look for
    ``serving_model`` and a class-name ordering the champion bundle does not use, and
    a shared check that passed for both would be checking almost nothing.
    """
    for required in ("model", "feature_columns", "model_version"):
        if required not in bundle:
            raise ServiceFault(f"The BWF champion bundle is missing '{required}'.")
    if bundle.get("stream") != "bwf":
        raise ServiceFault("The BWF champion bundle does not declare the bwf stream.")
    model = bundle["model"]
    if not hasattr(model, "decision_function") or not hasattr(model, "roles"):
        raise ServiceFault(
            "The BWF champion bundle's model is not a symmetric estimator; order "
            "symmetry could not be guaranteed."
        )


def _validate_glicko_state(state: dict[str, Any]) -> None:
    for required in ("players", "snapshot_as_of", "feature_code_version"):
        if required not in state:
            raise ServiceFault(f"The Glicko state snapshot is missing '{required}'.")
    if not state["players"]:
        raise ServiceFault("The Glicko state snapshot contains no rated players.")


def _validate_bundle(bundle: dict[str, Any], expected_stream: str) -> None:
    required = {
        "stream",
        "champion_name",
        "base_model_for_explanation",
        "feature_columns",
        "class_names",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ServiceFault(
            f"{expected_stream} model bundle is missing keys: {missing}"
        )
    if bundle["stream"] != expected_stream:
        raise ServiceFault(
            f"Expected {expected_stream!r} bundle, received {bundle['stream']!r}"
        )
    if "serving_model" not in bundle and "calibrated_model" not in bundle:
        raise ServiceFault(f"{expected_stream} model bundle has no serving model")
    _check_class_order(bundle, expected_stream)


def clear_model_caches() -> None:
    load_bwf_champion_bundle.cache_clear()
    load_bwf_glicko_state.cache_clear()
    load_bwf_bundle.cache_clear()
    load_injury_bundle.cache_clear()
    load_injury_binary_bundle.cache_clear()
    load_bwf_state.cache_clear()


def _artifact_paths() -> dict[str, Path]:
    """The artefacts the serving caches are built from, by cache name."""
    models = PROJECT_ROOT / "models"
    baseline_dir = models / "baseline"
    champion_dir = models / "champion"
    bwf_refit = baseline_dir / "bwf_baseline_bundle.joblib"
    injury_refit = champion_dir / "injury_champion_bundle.joblib"
    return {
        "bwf_bundle": bwf_refit if bwf_refit.exists() else models / "bwf_champion_bundle.joblib",
        "injury_bundle": (
            injury_refit if injury_refit.exists() else models / "injury_champion_bundle.joblib"
        ),
        "injury_binary_bundle": INJURY_BINARY_BUNDLE,
        "bwf_state": PROJECT_ROOT / "artifacts" / "bwf_state_snapshot.joblib",
        "bwf_champion_bundle": champion_dir / "bwf_champion_bundle.joblib",
        "bwf_glicko_state": PROJECT_ROOT / "artifacts" / "bwf_glicko_state_snapshot.joblib",
    }


def artifact_fingerprints() -> dict[str, dict[str, Any]]:
    """SHA-256 and size of each serving artefact, for reporting what changed.

    Hashed rather than compared by modification time: a redeployed file can carry
    any timestamp, and a file restored from a backup can look older than the one
    it replaced. The hash is what the freeze manifest records, so this reports
    identity in the same terms the governance trail already uses.
    """
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in _artifact_paths().items():
        if not path.exists():
            fingerprints[name] = {"path": str(path), "present": False}
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        fingerprints[name] = {
            "path": str(path),
            "present": True,
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    return fingerprints


def reload_model_caches() -> dict[str, Any]:
    """Re-read the serving artefacts from disk, but only if they are usable.

    The obvious implementation - clear the caches and let the next request load
    whatever is on disk - fails in the one situation the endpoint exists for. A
    reload happens right after a file has been replaced, which is exactly when
    that file is most likely to be truncated, half-copied or exported wrong; and
    clearing first means the damage is discovered by the next athlete's request,
    with the working model already discarded and no way back without a restart.

    So the candidates are loaded and validated *before* anything is dropped. If
    every artefact loads and passes the same checks the cached ones passed, the
    caches are cleared and the new files take effect. If any fails, the caches
    are left untouched, the service carries on answering from the models it
    already has, and the error is reported. A refused reload is a working
    service, which is the outcome to prefer.
    """
    before = artifact_fingerprints()

    # Validate candidates without disturbing the live caches. These calls
    # deliberately bypass the lru_cache wrappers by re-running the load logic.
    problems: list[str] = []
    for name, path in _artifact_paths().items():
        try:
            if name == "bwf_bundle":
                _validate_bundle(_load(path, "The BWF model bundle"), "bwf")
            elif name == "injury_bundle":
                _validate_bundle(_load(path, "The injury model bundle"), "injury")
            elif name == "injury_binary_bundle":
                # The binary artefact is a research candidate on its own routes.
                # Its absence must not refuse a reload of the served models.
                if path.exists():
                    _validate_bundle(
                        _load(path, "The binary injury model bundle"), "injury_binary"
                    )
            elif name == "bwf_champion_bundle":
                _validate_champion_bundle(_load(path, "The BWF champion model bundle"))
            elif name == "bwf_glicko_state":
                _validate_glicko_state(_load(path, "The BWF Glicko-2 state snapshot"))
            else:
                _validate_state(_load(path, "The BWF rating state snapshot"))
        except (ArtifactUnavailable, ServiceFault) as error:
            problems.append(f"{name}: {error}")

    if problems:
        return {
            "reloaded": False,
            "reason": (
                "The artefacts on disk did not load cleanly, so the running "
                "models were kept and nothing was replaced."
            ),
            "problems": problems,
            "artifacts": before,
            "changed": [],
        }

    clear_model_caches()
    after = artifact_fingerprints()
    changed = [
        name
        for name in after
        if before.get(name, {}).get("sha256") != after[name].get("sha256")
    ]
    return {
        "reloaded": True,
        "reason": (
            "Every artefact loaded and validated, so the caches were cleared and "
            "the next request will use the files now on disk."
        ),
        "problems": [],
        "artifacts": after,
        "changed": changed,
    }
