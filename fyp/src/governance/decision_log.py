"""Server-side, append-only audit trail for every prediction the API serves."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import threading

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_LOG_ROOT = PROJECT_ROOT / "outputs" / "governance" / "decision_log"


def _test_runner() -> str | None:
    """The test runner driving this process, if one is.

    Deliberately narrow. ``"unittest" in sys.modules`` is not evidence of a
    test run - ``unittest.mock`` is imported by ordinary libraries - so the
    check is on how the process was *started*: ``python -m unittest`` sets
    ``__main__.__spec__.name`` to ``unittest.__main__``, and pytest is only in
    ``sys.modules`` when pytest is what is running.

    This is what makes the redirection hold under every invocation style. The
    project's own documented command is ``python3 -m unittest discover -s
    tests``, which imports test modules top-level and therefore never executes
    ``tests/__init__.py`` - so a guard that lives only in that file is a guard
    that is off exactly when the documented command is used. That is how eight
    further records reached the log on 2026-08-20 while this correction was
    being written.
    """
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    name = getattr(spec, "name", "") or ""
    if name == "unittest" or name.startswith("unittest."):
        return "unittest"
    if "pytest" in sys.modules:
        return "pytest"
    return None


_RUNNER = _test_runner()

_OVERRIDE = os.environ.get("FYP_DECISION_LOG_ROOT")
_OWNED_TEMPORARY_ROOT = (
    tempfile.mkdtemp(prefix="fyp-automated-decision-log-")
    if _RUNNER and not _OVERRIDE
    else None
)
DECISION_LOG_ROOT = Path(_OVERRIDE or _OWNED_TEMPORARY_ROOT or PRODUCTION_LOG_ROOT)

TEST_PURPOSE = os.environ.get("FYP_DECISION_LOG_PURPOSE") or (
    f"automated_test:{_RUNNER}" if _RUNNER else None
)

if _OWNED_TEMPORARY_ROOT:
    atexit.register(shutil.rmtree, _OWNED_TEMPORARY_ROOT, ignore_errors=True)

MAX_LINE_BYTES = 64 * 1024

_WRITE_LOCK = threading.Lock()

_FLAG_FIELDS = ("feature", "severity", "reason")


def build_decision_record(*, stream: str, model_version: str, prediction: str,
                          action: str, reason: str, input_payload: dict) -> dict:
    """Create a privacy-minimised, user-downloadable prototype audit record.

    This is the *human* half of the trail - what the model said and what the
    reviewer decided to do about it - and it is built when a person acts.
    ``build_served_record`` below is the machine half, written whenever the API
    answers, whether or not anybody ever reviews the result.
    """
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stream": stream,
        "model_version": model_version,
        "model_prediction": prediction,
        "user_action": action,
        "override_reason": reason.strip(),
        "input_fingerprint_sha256": input_fingerprint(input_payload),
        "raw_inputs_stored": False,
        "prototype_note": "In-session FYP evidence only; production requires authentication, retention and access controls.",
    }


def input_fingerprint(payload: Any) -> str:
    """SHA-256 of the inputs, computed the same way the dashboard computes it.

    ``sort_keys`` is what makes the digest a fingerprint rather than a hash of
    a particular serialisation: two callers who send the same session with
    their keys in a different order must land on the same line of evidence, or
    the log cannot answer whether a session was scored twice. ``default=str``
    keeps it total - a numpy scalar or a timestamp that slipped through is
    stringified rather than raising and costing the caller their audit record.
    """
    return sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _safe_flags(flags: Any) -> list[dict[str, Any]]:
    """Out-of-distribution flags reduced to name, severity and reason."""
    if not isinstance(flags, Iterable) or isinstance(flags, (str, bytes, dict)):
        return []
    reduced = []
    for flag in flags:
        if isinstance(flag, dict):
            reduced.append({key: flag.get(key) for key in _FLAG_FIELDS})
    return reduced


def _safe_conformal(conformal: Any) -> dict[str, Any] | None:
    """The conformal verdict, as verdict rather than as internals.

    Only the fields a reviewer would cite are kept, so a future addition to the
    conformal payload cannot enlarge the audit record without somebody choosing
    to record it.
    """
    if not isinstance(conformal, dict):
        return None
    return {
        "prediction_set": list(conformal.get("prediction_set") or []),
        "set_size": conformal.get("set_size"),
        "confident": conformal.get("confident"),
        "call": conformal.get("call"),
        "status": conformal.get("status"),
        # The coverage level and the method that produced the set. Without
        # them the record says "declined" without saying declined at what
        # confidence, under which calibration - which is not reviewable.
        "target_coverage": conformal.get("target_coverage"),
        "method": conformal.get("method"),
    }


def build_served_record(
    *,
    stream: str,
    endpoint: str,
    inputs: Any,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """The record written for one served prediction.

    ``inputs`` is consumed only to produce the fingerprint and is never stored;
    see the module docstring for why. ``prediction`` is the response body the
    serving layer built, so the record quotes what the caller was actually told
    rather than a recomputed approximation of it.
    """
    conformal = _safe_conformal(prediction.get("conformal"))
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stream": stream,
        "endpoint": endpoint,
        "input_fingerprint_sha256": input_fingerprint(inputs),
        "raw_inputs_stored": False,
        "model_version": prediction.get("model_version"),
        "model_prediction": prediction.get("prediction"),
        "probabilities": prediction.get("probabilities") or {},
        "conformal": conformal,
        "conformal_verdict": (
            None
            if conformal is None
            else ("confident" if conformal.get("confident") else "declined")
        ),
        "uncertainty_status": prediction.get("uncertainty_status"),
        "out_of_distribution_flags": _safe_flags(
            prediction.get("out_of_distribution_flags")
        ),
        "host_country_known": prediction.get("host_country_known"),
        "human_review_required": bool(prediction.get("human_review_required", True)),
        # Present only when the caller declared itself automated. Absent on a
        # real served prediction, so its presence is evidence rather than a
        # default that could be read as one.
        **({"test_purpose": TEST_PURPOSE} if TEST_PURPOSE else {}),
    }


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    """A record stripped to its identifying fields, for the size bound.

    Reached only if the full record would breach ``MAX_LINE_BYTES`` - a model
    with an implausible number of classes, say. Writing something that still
    identifies the request and admits what was dropped beats writing an
    oversized line whose atomicity nobody can vouch for, and beats writing
    nothing at all.
    """
    keep = (
        "schema_version",
        "recorded_at_utc",
        "stream",
        "endpoint",
        "input_fingerprint_sha256",
        "raw_inputs_stored",
        "model_version",
        "model_prediction",
        "conformal_verdict",
        "human_review_required",
    )
    reduced = {key: record.get(key) for key in keep}
    reduced["truncated"] = True
    reduced["truncation_reason"] = (
        f"The full record exceeded the {MAX_LINE_BYTES}-byte line bound that "
        "keeps a single-write append atomic, so detail fields were dropped."
    )
    return reduced


def log_path(root: Path | str | None = None, moment: datetime | None = None) -> Path:
    """The file one record belongs in: one per UTC date, under ``root``.

    Partitioning by date is what makes the log usable without a database. A
    reviewer asking "what did the service serve on the 12th?" opens one file,
    and a run that ends up serving a great many predictions does not produce a
    single file too large to read.
    """
    base = Path(root) if root is not None else DECISION_LOG_ROOT
    when = moment or datetime.now(timezone.utc)
    return base / f"{when.astimezone(timezone.utc).date().isoformat()}.jsonl"


def append_record(record: dict[str, Any], root: Path | str | None = None) -> dict[str, Any]:
    """Append one record, and report honestly whether it landed.

    Returns the ``audit`` block the API attaches to its response. This function
    raises nothing: every failure path a filesystem can produce - a read-only
    volume, a full disk, a path that is a directory, a record carrying
    something json cannot encode - ends as ``recorded: false`` with a reason,
    because none of them is a reason to withhold a prediction that was
    correctly computed.
    """
    path = log_path(root)
    try:
        line = json.dumps(record, default=str, ensure_ascii=False)
        payload = (line + "\n").encode("utf-8")
        if len(payload) > MAX_LINE_BYTES:
            payload = (
                json.dumps(_compact(record), default=str, ensure_ascii=False) + "\n"
            ).encode("utf-8")

        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND, and exactly one write() for the whole line. The kernel
        # advances the offset and writes under a single atomic step for a
        # regular file, so a concurrent appender cannot land inside this line.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            with _WRITE_LOCK:
                os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError) as error:
        # Deliberately noisy on the server and honest to the caller. The one
        # outcome this must never have is a quiet one.
        logger.error(
            "Decision record for stream %r could not be written to %s: %s",
            record.get("stream"),
            path,
            error,
            exc_info=error,
        )
        return {
            "recorded": False,
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
            "message": (
                "The prediction was served but its server-side audit record "
                "could not be written, so this result is not in the decision "
                "log. Treat it as unaudited and report the failure."
            ),
        }

    return {
        "recorded": True,
        "path": str(path),
        "input_fingerprint_sha256": record.get("input_fingerprint_sha256"),
        "recorded_at_utc": record.get("recorded_at_utc"),
        "raw_inputs_stored": False,
    }


def record_decision(
    *,
    stream: str,
    endpoint: str,
    inputs: Any,
    prediction: dict[str, Any],
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Build and append a record for one served prediction, never raising.

    The single entry point the API uses. Record construction is inside the
    guard as well as the write, because a prediction body shaped unexpectedly
    must not be able to fail a request either.
    """
    try:
        record = build_served_record(
            stream=stream, endpoint=endpoint, inputs=inputs, prediction=prediction
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.error("Decision record could not be built: %s", error, exc_info=error)
        return {
            "recorded": False,
            "error": f"{type(error).__name__}: {error}",
            "message": (
                "The prediction was served but no audit record could be built "
                "for it. Treat this result as unaudited."
            ),
        }
    return append_record(record, root=root)


# ------------------------------------------------------------------ read path
# A log nobody can read is not evidence. These functions answer the questions an
# examiner or supervisor actually asks of an audit trail: how much was served,
# by which model version, and how often the model declined to call it.


def read_records(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Every readable record, oldest file first.

    A line that cannot be parsed is skipped rather than fatal. A truncated
    final line - the one shape an append-only file can legitimately be caught
    in - must not make the preceding thousand records unreadable.
    """
    base = Path(root) if root is not None else DECISION_LOG_ROOT
    if not base.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Decision log %s could not be read: %s", path, error)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping an unreadable line in %s", path)
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


CORRECTION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "governance"
    / "decision_log_corrections"
    / "automated_run_exclusions.json"
)


def load_corrections(path: Path | str | None = None) -> dict[str, Any]:
    """The audit-correction manifest, or an empty correction if there is none.

    An absent manifest means nothing has been proven synthetic, which is a
    different statement from "nothing is" and is reported as such: the summary
    below says how many rows it excluded, so zero exclusions is visible rather
    than assumed.
    """
    manifest_path = Path(path) if path is not None else CORRECTION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"excluded_row_sha256": set(), "manifest": None}
    hashes = {
        str(entry.get("row_sha256"))
        for entry in manifest.get("excluded_rows", [])
        if entry.get("row_sha256")
    }
    return {"excluded_row_sha256": hashes, "manifest": manifest}


def read_records_with_hashes(
    root: Path | str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Every readable record paired with the SHA-256 of the line it came from.

    The hash is over the bytes as written, which is what makes an exclusion
    checkable: a reviewer can hash a line themselves and find it in the
    correction manifest, and a rewritten line would no longer match - so the
    correction cannot be used to quietly disown a record that was later edited.
    """
    base = Path(root) if root is not None else DECISION_LOG_ROOT
    if not base.exists():
        return []
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Decision log %s could not be read: %s", path, error)
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping an unreadable line in %s", path)
                continue
            if isinstance(parsed, dict):
                rows.append((sha256(stripped.encode("utf-8")).hexdigest(), parsed))
    return rows


def _tally(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "unknown" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarise_decisions(
    root: Path | str | None = None, corrections: Path | str | None = None
) -> dict[str, Any]:
    """Counts by stream, verdict, model version and endpoint, plus the time range.

    Counts only. The summary carries no fingerprints, so it can be read aloud
    in a demonstration without exposing even the pseudonymous identifier, and
    it says plainly that raw inputs were never stored so a reader does not have
    to take the docstring's word for it.

    Records proven to have been written by an automated caller are excluded,
    and only those: the exclusion set is the row hashes in the audit-correction
    manifest, never a time range or an endpoint pattern. A row nobody has
    proven synthetic is counted, however plausible a candidate it looks. The
    log is not touched - it stays append-only - so the correction is
    subtractive at read time and the summary reports how much it subtracted.
    """
    rows = read_records_with_hashes(root)
    correction = load_corrections(corrections)
    excluded_hashes = correction["excluded_row_sha256"]
    kept = [record for row_hash, record in rows if row_hash not in excluded_hashes]
    excluded = len(rows) - len(kept)
    records = kept
    timestamps = sorted(
        str(record.get("recorded_at_utc"))
        for record in records
        if record.get("recorded_at_utc")
    )
    base = Path(root) if root is not None else DECISION_LOG_ROOT
    manifest = correction["manifest"]
    return {
        "log_directory": str(base),
        "total_records": len(records),
        "by_stream": _tally(record.get("stream") for record in records),
        "by_conformal_verdict": _tally(
            record.get("conformal_verdict") for record in records
        ),
        "by_model_version": _tally(record.get("model_version") for record in records),
        "by_endpoint": _tally(record.get("endpoint") for record in records),
        "by_prediction": _tally(record.get("model_prediction") for record in records),
        "distinct_input_fingerprints": len(
            {
                record.get("input_fingerprint_sha256")
                for record in records
                if record.get("input_fingerprint_sha256")
            }
        ),
        "first_recorded_at_utc": timestamps[0] if timestamps else None,
        "last_recorded_at_utc": timestamps[-1] if timestamps else None,
        "raw_inputs_stored": False,
        "audit_correction": {
            "records_written": len(rows),
            "records_excluded": excluded,
            "manifest": (
                str(CORRECTION_MANIFEST.relative_to(PROJECT_ROOT))
                if manifest
                else None
            ),
            "method": (manifest or {}).get("method"),
            "log_modified": False,
            "note": (
                "Excluded records were proven to have been written by an "
                "automated caller and are named by row hash in the manifest. "
                "The log itself is unchanged: nothing was deleted or rewritten."
            )
            if excluded
            else (
                "No record has been proven synthetic, so every record in the "
                "log is counted here."
            ),
        },
        "note": (
            "Append-only JSONL, one file per UTC date. Inputs are held as "
            "SHA-256 fingerprints only; no raw physiological or training "
            "values are stored. Movement screening is measured against "
            "declared policies rather than predicted, so it is not logged here."
        ),
    }
