"""The tournament calendar, and the optional live draw for a chosen tournament."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import os
import time
import uuid

import pandas as pd

from .data_registry import (
    PROJECT_ROOT,
    load_bwf_context_options,
    load_bwf_player_lookup,
    load_bwf_tournament_calendar,
)

API_BASE = "https://extranet-lv.bwfbadminton.com/api"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
DISCIPLINES = {"MS", "WS", "MD", "WD", "XD"}
DOUBLES = {"MD", "WD", "XD"}

LIVE_DRAW_ENV = "BWF_LIVE_DRAW"
_ENABLED_VALUES = {"on", "1", "true", "yes"}

DRAW_IS_NOT_A_MODEL_INPUT = (
    "The draw is an interface convenience for choosing a fixture without typing "
    "player names. It is never a model input: the forecast is computed from the "
    "frozen rating state and the frozen served bundle, and is identical whether the "
    "players were chosen from a draw or entered by hand."
)

NEXT_ROUND_UNPUBLISHED_MESSAGE = (
    "The latest BWF draw does not yet contain confirmed upcoming fixtures for "
    "this discipline. Refresh when the next-round draw is published, or choose "
    "the two players yourself below."
)

EVENT_FINISHED_MESSAGE = (
    "This event has finished, so it has no upcoming fixtures. Choose another "
    "event above, or choose the two players yourself below."
)

DRAW_DISABLED_MESSAGE = (
    "Live draw retrieval is switched off, so this tournament's fixtures were "
    "not requested from BWF. Choose the two players yourself below — the "
    "forecast is exactly the same either way. To enable the draw, set "
    f"{LIVE_DRAW_ENV}=on before starting the model service."
)

STALE_DRAW_MESSAGE = (
    "This draw is a copy read from BWF at {read_at} ({age} ago), and results "
    "since then are not in it. Fixtures dated on or before today cannot be "
    "confirmed as still unplayed from a copy this old, so they are held back "
    "rather than shown."
)
STALE_DRAW_LIVE_OFF_SUFFIX = (
    f" Live retrieval is switched off, so the copy will not refresh: set "
    f"{LIVE_DRAW_ENV}=on before starting the model service."
)
STALE_DRAW_LIVE_ON_SUFFIX = (
    " BWF could not be reached for a fresher copy; use \"Check BWF again\"."
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "bwf_draws"
# A draw changes at the pace of a tournament schedule, not a page refresh, and
# the BWF API rate limits bursts hard enough to answer 403. Ten minutes keeps
# the dashboard responsive without hammering an endpoint we do not own.
CACHE_TTL_SECONDS = 600


class DrawUnavailable(RuntimeError):
    """The draw could not be fetched and no cached copy could stand in."""


def _session():
    """Build the HTTP session, importing requests only when a fetch happens.

    Everything except the live draw works from artefacts on disk, so a missing
    or broken requests install should not stop the calendar being served.
    """
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            # Without this the API answers 403.
            "Referer": "https://bwfbadminton.com/",
        }
    )
    return session


def snapshot_date() -> pd.Timestamp:
    options = load_bwf_context_options()
    return pd.Timestamp(options["snapshot_as_of"]).normalize()


def first_predictable_date() -> pd.Timestamp:
    """The earliest date the model will accept: the day after the snapshot."""
    return snapshot_date() + pd.Timedelta(days=1)


def _now_utc() -> pd.Timestamp:
    """The current instant, in UTC.

    Its own function so a test can pin it. Everything else about this module is
    reproducible from artefacts on disk; the one genuinely time-dependent
    judgement — whether a fixture has already begun — is made here.
    """
    return pd.Timestamp.now(tz="UTC")


def _end_timestamp(entry: Any) -> pd.Timestamp | None:
    """The day the event finishes, or None where the export does not say.

    An unreadable end date leaves the event listed: the picker's job is to
    offer events, and dropping one because a date could not be parsed is a
    worse failure than showing an event a day too long.
    """
    if not isinstance(entry, dict):
        return None
    end = str(entry.get("end_date") or "").strip()
    if not end:
        return None
    try:
        return pd.Timestamp(end).normalize()
    except (ValueError, TypeError):
        return None


def _start_timestamp(entry: Any) -> pd.Timestamp | None:
    """The entry's start date, or ``None`` when it cannot be read as one.

    The calendar is re-exported from an upstream feed, so a single entry with a
    blank, "TBC" or otherwise unparseable start date is a data condition rather
    than a programming error. Letting ``pd.Timestamp`` raise turned one bad row
    into a 500 for the whole picker.
    """
    if not isinstance(entry, dict):
        return None
    start = str(entry.get("start_date") or "").strip()
    if not start:
        return None
    try:
        return pd.Timestamp(start)
    except (ValueError, TypeError):
        return None


def list_tournaments(
    *, upcoming_only: bool = True, limit: int | None = None
) -> dict[str, Any]:
    """The tournaments the dashboard may offer, newest commitments first.

    Entries whose start date cannot be read are reported separately rather than
    served with an unknown date or allowed to fail the whole request: a picker
    that silently drops an event is as misleading as one that crashes.
    """
    calendar = load_bwf_tournament_calendar()
    known_countries = set(load_bwf_context_options().get("host_countries") or [])
    cutoff = snapshot_date()
    # An event is dropped once its last day is behind us, not once its final
    # has been played. The two are different moments, and only the first is
    # knowable here: whether the final is over is a fact about the draw, which
    # this listing never reads and which is unavailable at all when live
    # retrieval is off or the cached copy is stale. Judging by the calendar
    # keeps the list truthful without pretending to know a result.
    today = _now_utc().normalize().tz_localize(None)

    entries: list[dict[str, Any]] = []
    undated = 0
    for entry in calendar.get("tournaments") or []:
        if not isinstance(entry, dict):
            undated += 1
            continue
        start = _start_timestamp(entry)
        if start is None:
            undated += 1
            continue
        if upcoming_only and start <= cutoff:
            continue
        end = _end_timestamp(entry)
        if upcoming_only and end is not None and end < today:
            continue
        host = entry.get("host_country")
        entries.append(
            {
                **entry,
                # The venue for a future event is sometimes still "TBC", and a
                # country absent from training is an unseen category rather than
                # a safe default, so both are surfaced instead of guessed at.
                "host_country_known": host in known_countries,
            }
        )

    # Sorted on the parsed timestamp rather than the raw string: the export is
    # ISO-formatted today, so the two orders agree, but a single differently
    # formatted date would silently reorder the whole calendar under a string
    # sort instead of being caught above.
    entries.sort(key=lambda item: _start_timestamp(item) or pd.Timestamp.max)
    if limit is not None:
        entries = entries[:limit]
    return {
        "snapshot_as_of": str(cutoff.date()),
        "tournament_count": len(entries),
        # Zero in normal operation. A non-zero count means the exported calendar
        # carries rows the dashboard cannot offer, and says so rather than
        # hiding them.
        "unreadable_entry_count": undated,
        "tournaments": entries,
    }


def find_tournament(tournament_id: int) -> dict[str, Any] | None:
    for entry in load_bwf_tournament_calendar().get("tournaments") or []:
        if not isinstance(entry, dict):
            continue
        try:
            entry_id = int(entry.get("tournament_id", -1))
        except (TypeError, ValueError):
            # An unreadable id cannot be the one being asked for. Skipping it
            # keeps one malformed row from hiding every tournament after it.
            continue
        if entry_id == int(tournament_id):
            return entry
    return None


def _flatten_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the two shapes the BWF match payload comes in, deduplicated.

    The same match appears under both the by-time and by-court groupings, so
    the match id decides which copy is kept.
    """
    results = payload.get("results")
    if not isinstance(results, dict):
        return []

    matches: list[dict[str, Any]] = []
    seen: set[Any] = set()

    def add(match: Any) -> None:
        if not isinstance(match, dict):
            return
        key = match.get("id")
        key = key if key is not None else id(match)
        if key in seen:
            return
        seen.add(key)
        matches.append(match)

    by_time = results.get("by_time")
    if isinstance(by_time, dict):
        for match in by_time.get("time_group") or []:
            add(match)

    by_court = results.get("by_court")
    if isinstance(by_court, dict):
        for section in by_court.values():
            if isinstance(section, dict):
                for match in section.values():
                    add(match)

    return matches


def _parse_discipline(draw_name: Any) -> tuple[str | None, bool]:
    text = str(draw_name or "").strip()
    if not text:
        return None, False
    head = text.split()[0].split("-")[0].strip().upper()
    if head not in DISCIPLINES:
        return None, False
    return head, "qual" in text.lower()


def _player(detail: Any, known_ids: set[int]) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    raw_id = detail.get("id")
    try:
        player_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    name = str(detail.get("name_display") or detail.get("name") or "").strip()
    if not name:
        return None
    return {
        "player_id": player_id,
        "player_name": name,
        "country": str(detail.get("nationality") or "").strip(),
        # A player the ratings have never seen still predicts, but from the
        # default rating rather than a real one, so the caller is told.
        "rated": player_id in known_ids,
    }


def _epoch_to_date(value: Any) -> tuple[str, str]:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return "", ""
    moment = pd.Timestamp(timestamp, unit="s", tz="UTC")
    return str(moment.date()), moment.isoformat()


def _fixture(
    match: dict[str, Any],
    known_ids: set[int],
    earliest: pd.Timestamp,
    fallback_date: str,
    *,
    now: pd.Timestamp | None = None,
    event_end_date: str | None = None,
    read_at: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """One BWF match row as a picker fixture, or None for a non-individual draw.

    ``read_at`` is the instant the draw was read from BWF. A copy older than
    ``CACHE_TTL_SECONDS`` is stale: its ``winner`` field says nothing about play
    since it was taken, so a fixture whose day is already here cannot be vouched
    for from it and is blocked (not hidden — the row stays visible with the
    reason on it).
    """
    discipline, is_qualification = _parse_discipline(match.get("draw_name"))
    if discipline is None:
        return None

    is_doubles = discipline in DOUBLES
    team1 = [
        player
        for player in (
            _player(match.get("t1p1_detail"), known_ids),
            _player(match.get("t1p2_detail"), known_ids) if is_doubles else None,
        )
        if player is not None
    ]
    team2 = [
        player
        for player in (
            _player(match.get("t2p1_detail"), known_ids),
            _player(match.get("t2p2_detail"), known_ids) if is_doubles else None,
        )
        if player is not None
    ]

    try:
        winner = int(match.get("winner") or 0)
    except (TypeError, ValueError):
        winner = 0

    match_date, match_datetime = _epoch_to_date(match.get("start_time"))
    if not match_date:
        # Early-round fixtures are often published before they are scheduled;
        # the tournament start date is the honest stand-in.
        match_date = fallback_date

    moment = _now_utc() if now is None else pd.Timestamp(now)
    moment = (
        moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    )
    today = moment.normalize().tz_localize(None)

    expected = 2 if is_doubles else 1
    blockers: list[str] = []
    if len(team1) != expected or len(team2) != expected:
        # A half-empty side is a fixture whose qualifier has not been decided.
        blockers.append("Both sides are not confirmed yet")

    # Whether the first serve is already behind us.
    #
    # This is a separate question from `winner`, and treating the two as one is
    # what let a played match reach the picker. The winner field is written when
    # BWF's own feed is updated, which is not the moment the match ends: a
    # quarter-final played yesterday can still arrive with `winner: 0` from a
    # feed that has not caught up, or from this module's own ten-minute cache.
    # A pre-match forecast for a match that has already been played is not a
    # forecast, so the clock is asked directly rather than inferred from a
    # result that may not have been published yet.
    if match_datetime:
        scheduled_at = pd.Timestamp(match_datetime)
        if scheduled_at == scheduled_at.normalize():
            # Midnight UTC exactly, which is how BWF publishes a fixture that
            # has a day but no time of day yet — every match of the World
            # Championships arrives stamped 00:00:00 on its date. Read as an
            # instant it says "this match began at midnight", which blocked
            # every fixture scheduled for today from the moment the day
            # started: on the morning of the semi-finals the picker offered
            # nothing at all. A date is therefore compared as a date, and only
            # a day already behind us counts as played.
            has_started = scheduled_at.normalize().tz_localize(None) < today
        else:
            # A real time of day is the exact boundary: a fixture stops being
            # forecastable when play is due to begin, not at the end of its day.
            has_started = scheduled_at <= moment
    else:
        # No start time was published, so the fixture's own date is the
        # tournament's start date rather than a schedule, and it is no evidence
        # at all about a later round. The only defensible statement about an
        # untimed fixture is that it cannot still be ahead of us once the whole
        # event is over.
        event_end = None
        if event_end_date:
            try:
                event_end = pd.Timestamp(event_end_date).normalize()
            except (ValueError, TypeError):
                event_end = None
        has_started = event_end is not None and event_end < today

    # Whether the copy of the draw this row came from is recent enough to
    # vouch that the match has not been played since. `winner` is only as
    # current as the copy: a draw read at 10:11 and served at 15:30 still says
    # `winner: 0` for a semi-final that finished at 13:00. A genuine time of
    # day still ahead of us needs no vouching, but a fixture stamped with only
    # a date that is today, or with no time at all, can only be offered when
    # the copy is fresh.
    unverifiable = False
    if read_at is not None and not has_started and not winner:
        read_moment = pd.Timestamp(read_at)
        read_moment = (
            read_moment.tz_localize("UTC")
            if read_moment.tzinfo is None
            else read_moment.tz_convert("UTC")
        )
        copy_is_stale = (moment - read_moment).total_seconds() > CACHE_TTL_SECONDS
        if copy_is_stale:
            if match_datetime:
                scheduled_at = pd.Timestamp(match_datetime)
                if scheduled_at == scheduled_at.normalize():
                    unverifiable = (
                        scheduled_at.normalize().tz_localize(None) <= today
                    )
                # else: a real time of day later than now is still ahead of us.
            else:
                # Untimed fixture at an event under way: nothing but the copy's
                # own age says whether it has been played.
                unverifiable = True

    # The snapshot bound is stated before the clock, because a fixture from
    # before the snapshot is refused by the model itself whereas one from last
    # week is merely too late to forecast, and only the first reason tells a
    # reader something they cannot see on the row already. `blocked_reason`
    # shows the first entry, so the order here is what the picker says.
    if match_date:
        try:
            scheduled = pd.Timestamp(match_date)
        except (ValueError, TypeError):
            # The fallback date comes from the exported calendar, which is a
            # feed rather than a frozen artefact. An unreadable date blocks
            # this one fixture and says so; it does not fail the draw.
            scheduled = None
            blockers.append("The scheduled date could not be read")
        if scheduled is not None and scheduled < earliest:
            blockers.append(
                f"Played before the rating snapshot ({str(earliest.date())} onwards only)"
            )

    if winner:
        blockers.append("Already played")
    elif has_started:
        blockers.append(
            f"Already started ({match_date})" if match_date else "Already started"
        )
    elif unverifiable:
        blockers.append(
            "May already have been played: the draw copy is too old to confirm"
        )

    return {
        "match_id": str(match.get("id") or ""),
        "discipline": discipline,
        "is_qualification": is_qualification,
        "round_name": str(match.get("round_name") or "").strip() or "Unknown",
        "match_date": match_date,
        "start_datetime_utc": match_datetime,
        # Three states rather than two, because "the result is in" and "play
        # has begun" are different facts and only the second is always knowable
        # at the moment it becomes true. Both are past, and neither is
        # forecastable.
        "status": (
            "completed" if winner else "started" if has_started else "scheduled"
        ),
        # True when the row is still "scheduled" as far as the copy knows, but
        # the copy is too old to say so with confidence. Distinct from the
        # status because it is a fact about the copy, not about the match.
        "unverifiable": unverifiable,
        "winner": winner,
        "team1": team1,
        "team2": team2,
        "predictable": not blockers,
        "blocked_reason": blockers[0] if blockers else None,
    }


def _cache_path(tournament_id: int) -> Path:
    return CACHE_DIR / f"{int(tournament_id)}.json"


def _read_cache(tournament_id: int) -> tuple[dict[str, Any] | None, float]:
    path = _cache_path(tournament_id)
    if not path.exists():
        return None, 0.0
    try:
        with path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None, 0.0
    return cached.get("payload"), float(cached.get("fetched_at") or 0.0)


def _write_cache(tournament_id: int, payload: dict[str, Any]) -> None:
    """Replace the cached draw atomically, one temp file per writer.

    Two dashboard tabs opening the same tournament fetch it concurrently, and a
    shared ``<id>.json.tmp`` meant both wrote into the same handle: the second
    writer's opening brace landed inside the first writer's document, and
    whichever ``replace`` ran last promoted that interleaved file over a cache
    entry that was previously good. A unique temp name per write makes each
    writer's document whole, so the loser of the race merely overwrites the
    winner with an equally valid copy.
    """
    path = _cache_path(tournament_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.json.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump({"fetched_at": time.time(), "payload": payload}, handle)
        temp.replace(path)
    finally:
        # Only reachable when the write or the replace failed; a successful
        # replace has already moved the file.
        if temp.exists():
            temp.unlink(missing_ok=True)


def live_draw_enabled() -> bool:
    """Whether this process may ask BWF for a draw.

    Off unless ``BWF_LIVE_DRAW`` says otherwise. Opt-in rather than opt-out: the
    draw is a convenience, the calendar and manual selection cover the same
    ground without it, and a default that reaches the network is a default that
    fails where there is none.
    """
    return os.environ.get(LIVE_DRAW_ENV, "off").strip().lower() in _ENABLED_VALUES


def fetch_draw_payload(
    tournament_id: int,
    *,
    max_age_seconds: int = CACHE_TTL_SECONDS,
    timeout: int = 20,
) -> tuple[dict[str, Any], bool, float]:
    """Return ``(payload, from_cache, fetched_at)`` for one tournament's draw.

    A fresh enough cache entry is used without touching the network. When the
    fetch fails, a stale cache entry is preferred over an error, because a draw
    from an hour ago is far more useful than nothing.

    With live retrieval disabled this never opens a socket: an already-cached
    draw is still served, at whatever age it has, and anything else raises
    ``DrawUnavailable`` for the caller to explain.
    """
    cached, fetched_at = _read_cache(tournament_id)
    if cached is not None and (time.time() - fetched_at) < max_age_seconds:
        return cached, True, fetched_at

    if not live_draw_enabled():
        if cached is not None:
            # Stale, but read from disk rather than fetched. Age travels with it.
            return cached, True, fetched_at
        raise DrawUnavailable(DRAW_DISABLED_MESSAGE)

    try:
        session = _session()
        response = session.get(
            f"{API_BASE}/vue-tournament-matches",
            params={
                "drawCount": 0,
                "searchKey": "",
                "tmtId": int(tournament_id),
                "tmtType": 0,
                "isPara": "false",
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            raise DrawUnavailable(
                f"BWF returned {response.status_code} for tournament "
                f"{tournament_id}."
            )
        payload = response.json()
    except DrawUnavailable:
        if cached is not None:
            return cached, True, fetched_at
        raise
    except Exception as error:  # network, JSON, or a missing requests install
        if cached is not None:
            return cached, True, fetched_at
        raise DrawUnavailable(
            f"Could not reach the BWF draw service: {error}"
        ) from error

    _write_cache(tournament_id, payload)
    return payload, False, time.time()


def tournament_draw(
    tournament_id: int,
    *,
    discipline: str | None = None,
    include_completed: bool = False,
    max_age_seconds: int = CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    """The fixtures for one tournament, ready for the fixture picker."""
    tournament = find_tournament(tournament_id)
    if tournament is None:
        raise KeyError(
            f"Tournament {tournament_id} is not in the exported calendar."
        )
    if discipline is not None and discipline not in DISCIPLINES:
        raise ValueError(f"Unknown discipline {discipline!r}.")

    try:
        payload, from_cache, fetched_at = fetch_draw_payload(
            tournament_id, max_age_seconds=max_age_seconds
        )
    except DrawUnavailable:
        if not live_draw_enabled():
            # Switched off is a normal, explainable state rather than a fault:
            # the same payload shape comes back with no fixtures and a sentence
            # saying what to do instead, so no caller has to special-case it.
            return _no_draw_payload(tournament, discipline, DRAW_DISABLED_MESSAGE)
        raise
    matches = _flatten_matches(payload)

    lookup = load_bwf_player_lookup()
    known_ids = set(lookup["player_id"].dropna().astype(int))
    earliest = first_predictable_date()

    # One reading of the clock for the whole draw, so two fixtures either side
    # of the current instant cannot be judged against different "now"s.
    now = _now_utc()
    today = now.normalize().tz_localize(None)
    read_at = (
        pd.Timestamp(fetched_at, unit="s", tz="UTC") if fetched_at else None
    )
    age_seconds = (
        max(0.0, (now - read_at).total_seconds()) if read_at is not None else None
    )
    stale = age_seconds is not None and age_seconds > CACHE_TTL_SECONDS

    fixtures: list[dict[str, Any]] = []
    # How many rows this discipline had at all, and how many were set aside as
    # already played. The difference between "this draw holds nothing for you"
    # and "everything in it has been played" is the difference between two very
    # different sentences on screen, and only these counts can tell them apart.
    considered = past = unconfirmed = 0
    for match in matches:
        fixture = _fixture(
            match,
            known_ids,
            earliest,
            str(tournament.get("start_date") or ""),
            now=now,
            event_end_date=str(tournament.get("end_date") or "") or None,
            read_at=read_at,
        )
        if fixture is None:
            continue
        if discipline is not None and fixture["discipline"] != discipline:
            continue
        considered += 1
        # Anything not still ahead of us is left out. The picker exists to
        # choose a match to forecast, and a match already under way or already
        # played is not one — offering it invites a "prediction" made with the
        # outcome available, which is the one thing this screen must not do.
        if not include_completed and fixture["status"] != "scheduled":
            past += 1
            continue
        # A fixture the copy is too old to vouch for is withheld for the same
        # reason, not listed with an explanation beside it. Shown, it was a row
        # a reader could neither use nor dismiss — the final of an event that
        # had almost certainly finished, sitting under a warning that said as
        # much. The honest form of "this may already have been played" is not
        # to offer the match at all.
        if not include_completed and fixture["unverifiable"]:
            unconfirmed += 1
            continue
        fixtures.append(fixture)

    fixtures.sort(
        key=lambda item: (
            item["start_datetime_utc"] or item["match_date"],
            item["round_name"],
        )
    )

    known_countries = set(load_bwf_context_options().get("host_countries") or [])
    predictable = sum(1 for fixture in fixtures if fixture["predictable"])
    return {
        "tournament": {
            **tournament,
            "host_country_known": tournament.get("host_country")
            in known_countries,
        },
        "discipline": discipline,
        "fixture_count": len(fixtures),
        "predictable_count": predictable,
        "draw_published": bool(matches),
        "considered_count": considered,
        "past_count": past,
        # Fixtures held back because this copy of the draw is too old to say
        # whether they have been played. Kept as a count rather than as rows,
        # so a caller reading the JSON can still tell "nothing left to play"
        # apart from "nothing this copy can vouch for".
        "unconfirmed_count": unconfirmed,
        # Three distinct states, each with its own sentence, because "nothing to
        # show" has three different causes here and only one of them is a
        # waiting game. Saying "no matches to predict" for all of them told a
        # reader on the morning of the semi-finals that there was nothing to
        # forecast, when in fact BWF had simply not published the next round to
        # this system yet.
        "message": _draw_message(tournament, matches, fixtures, past, today),
        "from_cache": from_cache,
        "fetched_at_utc": (
            pd.Timestamp(fetched_at, unit="s", tz="UTC").isoformat()
            if fetched_at
            else None
        ),
        "live_draw_enabled": live_draw_enabled(),
        # A copy older than the cache TTL. Said as its own flag, with the
        # sentence to show, because `from_cache` alone is true for a copy
        # nine minutes old that is perfectly good.
        "stale": stale,
        "age_seconds": age_seconds,
        "stale_message": (
            _stale_message(read_at, age_seconds) if stale else None
        ),
        "note": DRAW_IS_NOT_A_MODEL_INPUT,
        "fixtures": fixtures,
    }


def _describe_age(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{max(minutes, 1)} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    return f"{hours // 24} days"


def _stale_message(read_at: pd.Timestamp, age_seconds: float) -> str:
    text = STALE_DRAW_MESSAGE.format(
        read_at=read_at.strftime("%H:%M UTC on %Y-%m-%d"),
        age=_describe_age(age_seconds),
    )
    return text + (
        STALE_DRAW_LIVE_ON_SUFFIX if live_draw_enabled() else STALE_DRAW_LIVE_OFF_SUFFIX
    )


def _draw_message(
    tournament: dict[str, Any],
    matches: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    past: int,
    today: pd.Timestamp,
) -> str | None:
    """What to say when the fixture list is empty, and nothing when it is not.

    Each branch is a different thing to do next, which is the only reason to
    word them differently: wait for BWF, pick another event, or enter the two
    players by hand.
    """
    if fixtures:
        return None
    if not matches:
        # Draws appear a day or two before play, so an empty result for an event
        # weeks away is the normal case and not a failure.
        return (
            "BWF has not published this draw yet. Draws usually appear a day or "
            "two before play starts."
        )
    end = None
    try:
        end = pd.Timestamp(str(tournament.get("end_date") or "")).normalize()
    except (ValueError, TypeError):
        end = None
    if end is not None and end < today:
        return EVENT_FINISHED_MESSAGE
    if past:
        return NEXT_ROUND_UNPUBLISHED_MESSAGE
    # Rows exist, none were set aside as played, and still nothing is offered:
    # the draw holds only fixtures whose sides are undecided, or none for this
    # discipline at all.
    return NEXT_ROUND_UNPUBLISHED_MESSAGE


def _no_draw_payload(
    tournament: dict[str, Any], discipline: str | None, message: str
) -> dict[str, Any]:
    """A draw answer with no fixtures in it, in the shape callers already read.

    Every key a published draw carries is present, so the picker renders the
    explanation through its existing "no fixtures" path instead of needing a
    second one. `draw_published` is false because, as far as this system is
    concerned, no draw was published to it.
    """
    known_countries = set(load_bwf_context_options().get("host_countries") or [])
    return {
        "tournament": {
            **tournament,
            "host_country_known": tournament.get("host_country") in known_countries,
        },
        "discipline": discipline,
        "fixture_count": 0,
        "predictable_count": 0,
        "draw_published": False,
        "message": message,
        "from_cache": False,
        "fetched_at_utc": None,
        "live_draw_enabled": live_draw_enabled(),
        "stale": False,
        "age_seconds": None,
        "stale_message": None,
        "note": DRAW_IS_NOT_A_MODEL_INPUT,
        "fixtures": [],
    }


def fixture_to_prediction_request(
    fixture: dict[str, Any], tournament: dict[str, Any]
) -> dict[str, Any]:
    """Turn a chosen fixture into the body ``/bwf/predict`` expects."""
    return {
        "team1_player_ids": [player["player_id"] for player in fixture["team1"]],
        "team2_player_ids": [player["player_id"] for player in fixture["team2"]],
        "discipline": fixture["discipline"],
        "prediction_date": fixture["match_date"],
        "tier_category": tournament["tier_category"],
        "round_name": fixture["round_name"],
        "is_qualification": bool(fixture["is_qualification"]),
        "host_country": tournament["host_country"],
    }


__all__ = [
    "DRAW_IS_NOT_A_MODEL_INPUT",
    "EVENT_FINISHED_MESSAGE",
    "NEXT_ROUND_UNPUBLISHED_MESSAGE",
    "DrawUnavailable",
    "find_tournament",
    "first_predictable_date",
    "fixture_to_prediction_request",
    "list_tournaments",
    "live_draw_enabled",
    "tournament_draw",
]
