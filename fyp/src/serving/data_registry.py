from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_injury_development() -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / "outputs" / "injury" / "injury_development.csv"
    )


@lru_cache(maxsize=1)
def load_bwf_model_rows() -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "bwf"
        / "bwf_model_rows_2022_2025.csv",
        parse_dates=["date"],
    )


@lru_cache(maxsize=1)
def load_bwf_player_lookup() -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / "artifacts" / "bwf_player_lookup.csv"
    )


@lru_cache(maxsize=1)
def load_bwf_context_options() -> dict:
    with (
        PROJECT_ROOT / "artifacts" / "bwf_context_options.json"
    ).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4)
def _read_tournament_calendar(path: Path, modified_at: float) -> dict:
    """Cached on the file's modification time, so a re-export invalidates it.

    ``modified_at`` is never read: it is part of the cache key and nothing else.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bwf_tournament_calendar() -> dict:
    """The World Tour calendar exported by export_bwf_tournament_calendar.py.

    Unlike the frozen model artefacts, this file is re-exported whenever the
    tour calendar moves — new events appear, draws approach, dates shift. A
    plain ``lru_cache`` therefore pinned whatever the calendar said when the
    service booted, and a long-running server kept serving a calendar that was
    hours out of date while the file on disk had the event the user was looking
    for. Keying the cache on the modification time keeps the read cheap and the
    answer current.

    Absent until the export script has been run, so the caller is told what to
    run rather than being handed a bare missing-file error.
    """
    path = PROJECT_ROOT / "artifacts" / "bwf_tournament_calendar.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run "
            "scripts/export_bwf_tournament_calendar.py to build it."
        )
    return _read_tournament_calendar(path, path.stat().st_mtime)
