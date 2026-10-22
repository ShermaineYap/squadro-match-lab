"""Upload one short rally; get the stroke type of every hit in it."""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.broadcast import bst_adapter, players, shuttle as shuttle_module
from src.broadcast.frozen import (
    DEVELOPMENT_FOOTAGE,
    DIAGNOSTIC_FINDINGS,
    DIAGNOSTIC_FOOTAGE,
    FOOT_PULSE_STATEMENT,
    FOOT_PULSE_TORSOS,
    FROZEN_ON,
    FROZEN_THRESHOLDS,
    FUSION_STATES_REJECTED_STATEMENT,
    HITTER_SIDE_DERIVED_STATEMENT,
    JUMP_EVIDENCE_REJECTED,
    JUMP_EVIDENCE_REJECTED_ON,
    JUMP_EVIDENCE_REJECTION_EVIDENCE,
    JUMP_EVIDENCE_REJECTION_STATEMENT,
    JUMP_WINDOW_SECONDS,
    MMPOSE_SUBSTITUTION_STATEMENT,
    PLAYER_BOX_STATEMENT,
    MODEL_SCORE_STATEMENT,
    RALLY_CONTRACT,
    RALLY_CONTRACT_STATEMENT,
    STROKE_MODEL_SCORE_THRESHOLD,
    THRESHOLDS_STATEMENT,
)
from src.broadcast.third_party import dependency_status, provenance

REPLAY_FRAMES = 16
REPLAY_WIDTH = 480
REPLAY_MAX_STROKES = 12

COVERAGE_MINIMUM = 0.5

# --------------------------------------------------------------------------
# The primary result: which stroke was played
# --------------------------------------------------------------------------

STROKE_TYPE_LABELS: dict[str, str] = {
    "smash": "Smash",
    "drop": "Drop",
    "clear": "Clear",
    "lift": "Lift",
    "drive": "Drive",
    "net shot": "Net shot",
    "block": "Block",
    "push": "Push",
    "net kill": "Net kill",
    "cross-court net": "Cross-court net",
    "short serve": "Short serve",
    "long serve": "Long serve",
    bst_adapter.NOT_DETERMINED: "Not determined",
}

STROKE_TYPE_GROUPS: dict[str, str] = {
    "smash": "smash",
    "drop": "drop",
    "clear": "clear",
    "lift": "lift",
    "net shot": "net play",
    "block": "net play",
    "push": "net play",
    "net kill": "net play",
    "cross-court net": "net play",
    "short serve": "serve",
    "long serve": "serve",
    "drive": "other",
    bst_adapter.NOT_DETERMINED: "not determined",
}

STROKE_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("smash", "Smash"),
    ("drop", "Drop"),
    ("clear", "Clear"),
    ("lift", "Lift"),
    ("net play", "Net play"),
    ("serve", "Serve"),
    ("other", "Other"),
    ("not determined", "Not determined"),
)

PRIMARY_TASK_STATEMENT = (
    "The primary result for each hit is a predicted stroke type from a pretrained "
    "multi-class classifier: one of twelve named types, or not determined. Hits come "
    "from reversals in the TrackNetV3 shuttle trajectory; BST_0 predicts the type from "
    "the temporal pose of both players and the shuttle's path through the window. There "
    "is no jump-smash class in this taxonomy — a jump smash is a way of playing a smash, "
    "not a separate stroke — so a smash is reported as a smash whatever the jump "
    "evidence did. Jump evidence is supplementary experimental information and never "
    "changes the stroke type."
)

PER_CLASS_PERFORMANCE_STATEMENT = (
    "A class existing in the classifier is not evidence that it is predicted correctly. "
    "No per-class accuracy, precision or recall has been measured for any of these "
    "twelve types, on this footage or any other, by this project or anyone else. "
    "Per-class performance can only be established with an independently labelled "
    "evaluation set; see docs/BROADCAST_EVALUATION_PROTOCOL.md."
)


def not_determined_reason(
    call: bst_adapter.StrokeCall, *, both_players_seen: float, shuttle_seen: float
) -> str | None:
    """Why this hit has no stroke type, or ``None`` if it has one.

    This is the existing validity gate, stated once. It is exactly the set of
    conditions that already produced ``undetermined`` — the frozen model-score
    threshold, an unknown class, a missing side, and the coverage floor — with
    no condition added and none removed; a test pins that
    ``stroke_type == "not determined"`` if and only if ``state ==
    "undetermined"``, so the two can never come apart.

    Note what it does **not** consult: the jump evidence. Whether the foot pulse
    fired, did not fire, or could not be measured has no bearing on whether a
    stroke type is reported, or on which one.

    Ordered so the reason given is the one a reader can act on. Coverage comes
    first because it is the gate that overrides everything else: a window where
    the shuttle was barely seen is one where the classifier's input was mostly
    absent, which is a better answer than "the score was low".
    """
    floor = int(COVERAGE_MINIMUM * 100)
    if both_players_seen < COVERAGE_MINIMUM:
        return (
            f"both players were tracked in only {round(both_players_seen * 100)}% of this "
            f"window, under the {floor}% the pipeline requires before it reads a stroke"
        )
    if shuttle_seen < COVERAGE_MINIMUM:
        return (
            f"the shuttle was seen in only {round(shuttle_seen * 100)}% of this window, "
            f"under the {floor}% the pipeline requires before it reads a stroke"
        )
    if call.model_score < STROKE_MODEL_SCORE_THRESHOLD:
        # Three decimals, not two. At two, a raw 0.4968 printed as "scored 0.50,
        # under the frozen 0.5 threshold", which reads as a contradiction and
        # made a correct gate decision look like a bug. The gate has always used
        # the raw float; only the sentence was rounding.
        return (
            f"the top class scored {call.model_score:.3f}, under the frozen "
            f"{STROKE_MODEL_SCORE_THRESHOLD} model-score threshold"
        )
    if call.stroke == "unknown":
        return "the classifier's own unknown class scored highest"
    if call.side is None:
        return "the classifier's top class did not name which player hit the shuttle"
    return None


def primary_stroke_type(
    call: bst_adapter.StrokeCall, *, both_players_seen: float, shuttle_seen: float
) -> str:
    """The stroke type to display: the classifier's class, or not determined.

    A smash is a smash. The foot-pulse indicator is not consulted here and
    cannot be: it is not an argument of this function.
    """
    if not_determined_reason(
        call, both_players_seen=both_players_seen, shuttle_seen=shuttle_seen
    ) is not None:
        return bst_adapter.NOT_DETERMINED
    return call.stroke


STATES = (
    "likely_jump_smash",
    "smash_no_detected_jump",
    "smash_jump_not_evaluated",
    "jump_overhead_other",
    "other_stroke",
    "undetermined",
)
STATE_LABELS = {
    "likely_jump_smash": "Likely jump smash",
    "smash_no_detected_jump": "Smash — foot-pulse indicator did not fire",
    "smash_jump_not_evaluated": "Smash — jump not evaluated",
    "jump_overhead_other": "Foot pulse with a non-smash overhead stroke",
    "other_stroke": "Other stroke",
    "undetermined": "Not determined",
}
STATE_MEANINGS = {
    "likely_jump_smash": (
        "The classifier's top class was smash above the frozen score threshold and a "
        "lowest-foot vertical pulse was measured at the hit. Two uncalibrated, "
        "unvalidated indicators agreeing. Not a confirmed jump smash."
    ),
    "smash_no_detected_jump": (
        "Classified as smash; the foot-pulse indicator was evaluated and did not reach "
        "its threshold. The indicator did not fire — this is not proof the player "
        "stayed on the ground."
    ),
    "smash_jump_not_evaluated": (
        "Classified as smash; the foot-pulse indicator could not be measured at all "
        "(hitter untracked, occluded or unidentified). No jump evidence either way."
    ),
    "jump_overhead_other": (
        "A foot pulse was measured, but the classified stroke was an overhead other "
        "than smash. Named for what was observed, not rounded up to a smash."
    ),
    "other_stroke": "Classified as a stroke that is not an overhead. No jump-smash claim.",
    "undetermined": (
        "Abstention. The model score was below the frozen threshold, the class or the "
        "side was unknown, or the players or the shuttle were seen in under half the "
        "window. No stroke and no jump is claimed."
    ),
}

BOUNDARY = {
    "experimental": True,
    "used_in_movement_result": False,
    "used_in_policy_grading": False,
    "written_to_audit_record": False,
    "accuracy_measured_on_this_footage": False,
    "accuracy_measured_on_any_footage": False,
    "independently_evaluated": False,
    "production_ready": False,
    "model_score_is_calibrated": False,
    "jump_evidence_is_image_space_only": True,
}
BOUNDARY_STATEMENT = (
    "Experimental broadcast stroke analysis using pretrained third-party models "
    "(BST_0 stroke classifier on ShuttleSet weights; TrackNetV3 shuttle tracking; "
    "YOLO11 pose). This module has not been validated: no accuracy, precision or "
    "recall has been measured for it on any footage, by this project or anyone else. "
    "The numbers beside each stroke are uncalibrated model scores, not probabilities "
    "of being right. Jump evidence is an image-space indicator, not a measured jump. "
    "Nothing here enters the movement result, the policy grades, the official results "
    "or the audit record, and none of it is production ready."
)

# --------------------------------------------------------------------------
# Jump evidence
# --------------------------------------------------------------------------


def _torso_and_lowest_foot(kp: np.ndarray) -> tuple[float | None, float | None]:
    """Torso length and the image-y of the lower ankle, for COCO-17 keypoints."""
    ls, rs, lh, rh, la, ra = kp[5], kp[6], kp[11], kp[12], kp[15], kp[16]
    if not (ls.any() and rs.any() and lh.any() and rh.any() and la.any() and ra.any()):
        return None, None
    torso = float(np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2))
    # Larger y is lower in the image; the lower foot is the one on the ground
    # while running and the one that has to leave it for a jump.
    return (torso if torso > 1 else None), float(max(la[1], ra[1]))


def jump_evidence(
    tracks: players.PlayerTracks, hit: int, side_index: int, fps: float
) -> dict[str, Any]:
    """Measure the hitter's **lowest-foot vertical pulse** around the hit.

    The indicator: the hitter's lower ankle rises by at least
    ``FOOT_PULSE_TORSOS`` above where it was before its highest point and comes
    back down after it, all inside ``+/-JUMP_WINDOW_SECONDS``. Requiring the
    return is what separates a jump from perspective drift - a player retreating
    up the court raises every landmark in the image and never lowers them again
    within the window - and the lowest foot is used because one foot is always
    down while running.

    **This is an image-space indicator, not a physical jump height.** It is a
    displacement in pixels, scaled by the hitter's own torso length in the same
    picture, with no court homography, no camera calibration and no correction
    for the hitter's distance from the camera. It is reported as evidence about
    a jump, never as a measurement of one.

    Three outcomes, kept distinct:

    * ``present: True``  - the pulse was measured and reached the threshold.
    * ``present: False`` - the pulse was measured and did not reach it. The
      indicator did not fire; that is not proof the player stayed down.
    * ``present: None``  - nothing could be measured. **Not** a negative result.
    """
    span = int(fps * JUMP_WINDOW_SECONDS)
    lo, hi = max(0, hit - span), min(len(tracks.keypoints) - 1, hit + span)
    samples = [_torso_and_lowest_foot(tracks.keypoints[f, side_index]) for f in range(lo, hi + 1)]
    valid = [(t, y) for t, y in samples if t is not None and y is not None]
    common = {
        "indicator": "lowest-foot vertical pulse (image space)",
        "measures_physical_jump_height": False,
        "threshold_torsos": FOOT_PULSE_TORSOS,
        "window_seconds": JUMP_WINDOW_SECONDS,
        "note": FOOT_PULSE_STATEMENT,
        # Rejected on diagnostic footage before evaluation began. The
        # measurement is still computed and still reported, because deleting a
        # failed experiment is not the same as recording that it failed - but
        # it carries its rejection everywhere it goes, so no reader and no
        # results table can pick it up as a finding.
        "rejected": JUMP_EVIDENCE_REJECTED,
        "rejected_on": JUMP_EVIDENCE_REJECTED_ON,
        "rejection": JUMP_EVIDENCE_REJECTION_STATEMENT,
    }
    if len(valid) < 6:
        return {
            **common,
            "present": None,
            "evaluated": False,
            "reason": "the hitter's ankles, hips or shoulders were not tracked around the hit",
        }
    scale = float(np.median([t for t, _ in valid]))
    ys = np.array([y for _, y in valid])
    peak = int(np.argmin(ys))  # highest point of the lowest foot
    if peak == 0 or peak == len(ys) - 1:
        return {
            **common,
            "present": False,
            "evaluated": True,
            "foot_pulse_torsos": 0.0,
            "hitter_torso_px": round(scale, 1),
            "tracked_frames": len(valid),
            "reason": "no rise-and-return inside the window",
        }
    before = float(ys[:peak].max())
    after = float(ys[peak + 1:].max())
    # The pulse is bounded by the lower of the two grounded references, so a
    # foot that rose and never came back scores only its return leg.
    rise = float((min(before, after) - ys[peak]) / scale)
    return {
        **common,
        "present": rise >= FOOT_PULSE_TORSOS,
        "evaluated": True,
        "foot_pulse_torsos": round(rise, 3),
        # The scale the threshold is applied at. For the far player on 720p
        # broadcast footage the torso is ~35 px, so the 0.10 threshold is
        # 3-4 px - the same order as keypoint jitter. Reported so a reader can
        # weigh the evidence rather than take the boolean on trust.
        "hitter_torso_px": round(scale, 1),
        "tracked_frames": len(valid),
    }


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def fuse(call: bst_adapter.StrokeCall, jump: dict[str, Any]) -> str:
    """Name the combination of stroke call and jump evidence for what it is.

    The only asymmetry that matters: a smash whose jump evidence is ``None`` is
    ``smash_jump_not_evaluated``, never ``smash_no_detected_jump``. A missing
    measurement is not a measurement of absence, and the two must not share a
    label - a reader counting "smashes without a jump" would otherwise be
    counting untracked frames as grounded smashes.
    """
    if (
        call.model_score < STROKE_MODEL_SCORE_THRESHOLD
        or call.stroke == "unknown"
        or call.side is None
    ):
        return "undetermined"
    present = jump.get("present")
    if call.stroke == "smash":
        if present is True:
            return "likely_jump_smash"
        if present is False:
            return "smash_no_detected_jump"
        return "smash_jump_not_evaluated"
    if present is True and call.stroke in bst_adapter.OVERHEAD_STROKES:
        return "jump_overhead_other"
    return "other_stroke"


# --------------------------------------------------------------------------
# Replays
# --------------------------------------------------------------------------


def _replay(video: Path, start: int, end: int) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames: list[str] = []
    try:
        wanted = end - start + 1
        step = max(1, math.ceil(wanted / REPLAY_FRAMES))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while index <= end and len(frames) < REPLAY_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            if (index - start) % step == 0:
                h, w = frame.shape[:2]
                if w > REPLAY_WIDTH:
                    frame = cv2.resize(frame, (REPLAY_WIDTH, max(1, int(h * REPLAY_WIDTH / w))), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 62])
                if ok:
                    frames.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode())
            index += 1
    finally:
        capture.release()
    return frames


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------

MEASURED_SECONDS_PER_FRAME: dict[str, float] = {
    "reading": 0.005,
    "players": 0.139,
    "shuttle": 0.687,
    "hits": 0.001,
    "strokes": 0.002,
    "replays": 0.007,
}

MEASURED_FIXED_SECONDS = 44.5


@dataclass(frozen=True)
class Stage:
    """One reported step of the pipeline.

    ``determinate`` says whether the backend can actually see inside the stage.
    Pose and replays are loops this module drives, so they can be counted frame
    by frame. Shuttle tracking is a subprocess that prints nothing this module
    parses, so nothing inside it can be measured - and a bar that invents a
    position for it would be reporting a number no one computed. Those stages
    are marked indeterminate and the interface shows activity rather than a
    fraction.
    """

    key: str
    label: str
    weight: float
    determinate: bool


STAGES: tuple[Stage, ...] = (
    Stage("reading", "Reading footage", 0.02, True),
    Stage("players", "Detecting players", 0.17, True),
    Stage("shuttle", "Tracking the shuttle", 0.77, False),
    Stage("hits", "Detecting hit moments", 0.01, True),
    Stage("strokes", "Classifying strokes", 0.01, True),
    Stage("replays", "Preparing replays", 0.02, True),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}
LONGEST_STAGE = max(STAGES, key=lambda stage: stage.weight).key


def estimated_remaining_seconds(stage_key: str, done_in_stage: float, frames: int) -> float:
    """Seconds still to run, from the measured per-frame costs.

    A point estimate from six constants measured on one machine and one clip.
    It is deliberately never shown as one: the interface widens it into a range
    and rounds it, because a per-second countdown from this would be false
    precision dressed as a measurement.
    """
    remaining = 0.0
    seen = False
    for stage in STAGES:
        per_frame = MEASURED_SECONDS_PER_FRAME.get(stage.key, 0.0)
        if stage.key == stage_key:
            seen = True
            remaining += per_frame * frames * max(0.0, 1.0 - done_in_stage)
        elif seen:
            remaining += per_frame * frames
    return remaining


@dataclass
class Progress:
    """A callback target so a job can report which stage it is in.

    Kept callable as ``progress(fraction, message)`` so existing callers are
    unaffected, but the pipeline drives it through :meth:`enter` and
    :meth:`advance`, which carry the stage identity, whether the stage can be
    measured, the elapsed time and an estimate of what is left.
    """

    report: Any = None
    frames: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _stage: str = field(default="", init=False)

    # -- the original interface, unchanged ---------------------------------

    def __call__(self, fraction: float, message: str) -> None:
        self._emit(fraction=fraction, message=message)

    # -- the stage interface -----------------------------------------------

    def enter(self, key: str) -> None:
        """Begin a stage. Emits its start fraction and its label."""
        self._stage = key
        self.advance(0.0)

    def advance(self, done: float = 0.0, total: float = 1.0) -> None:
        """Report progress *within* the current stage, as a share of it."""
        stage = STAGE_BY_KEY.get(self._stage)
        if stage is None:
            return
        share = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
        before = sum(s.weight for s in STAGES if STAGES.index(s) < STAGES.index(stage))
        self._emit(
            fraction=before + stage.weight * share,
            message=stage.label,
            stage=stage,
            share=share,
        )

    def _emit(
        self,
        *,
        fraction: float,
        message: str,
        stage: Stage | None = None,
        share: float = 0.0,
    ) -> None:
        if self.report is None:
            return
        elapsed = time.monotonic() - self.started_at
        update: dict[str, Any] = {
            "progress": min(1.0, max(0.0, fraction)),
            "message": message,
            "elapsed_seconds": round(elapsed, 1),
        }
        if stage is not None:
            update["stage"] = stage.key
            update["stage_label"] = stage.label
            update["stage_determinate"] = stage.determinate
            update["stage_index"] = STAGES.index(stage) + 1
            update["stage_count"] = len(STAGES)
            if self.frames > 0:
                remaining = estimated_remaining_seconds(stage.key, share, self.frames)
                # A range, not a countdown. The band is wide because six
                # constants from one machine do not support a tighter one.
                update["estimated_remaining_seconds"] = {
                    "low": round(remaining * 0.7),
                    "high": round(remaining * 1.6),
                }
        self.report(update)


def stage_plan(frames: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "key": stage.key,
            "label": stage.label,
            "share_of_run": stage.weight,
            "measurable": stage.determinate,
            "longest": stage.key == LONGEST_STAGE,
            "estimated_seconds": (
                round(MEASURED_SECONDS_PER_FRAME.get(stage.key, 0.0) * frames)
                if frames > 0
                else None
            ),
        }
        for stage in STAGES
    ]


def analyse_broadcast_clip(
    video: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress: Progress | None = None,
    with_replays: bool = True,
    oracle_hits: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Analyse one rally. ``oracle_hits`` is an **evaluation-only** override.

    Normally the hits are the shuttle track's vertical reversals, and that is
    the only path the API, the job store and the dashboard can reach: nothing
    outside ``scripts/evaluate_broadcast_strokes.py`` and the tests passes
    ``oracle_hits``, and no request field maps onto it.

    It exists because the classifier's window runs from the *previous* hit to
    the *next* one (``bst_adapter.between_hits_windows``), so one spurious or
    missed hit corrupts the input to as many as three classifications. Measured
    end to end, a stroke-classification figure therefore carries an unknown
    amount of hit-detection error, and a low number is unattributable between
    the two stages. Supplying manually annotated contact frames isolates the
    classifier: the same weights, the same gate, the same thresholds, over
    windows that are known to be right.

    ``oracle_hits`` are **source** frame numbers, counted from the start of the
    uploaded file, exactly as they are annotated and exactly as
    ``shuttle.hit_frames`` reports them. They are converted to range-local
    indices here, and any that fall outside the analysed range are dropped.
    Passing them changes which windows are classified and nothing else: no
    threshold, no weight and no gate differs between the two paths, and the
    payload records which one produced it under ``shuttle.hit_source``.
    """
    progress = progress or Progress()
    status = dependency_status()
    if not status["ready"]:
        raise RuntimeError(status.get("install_hint", "Broadcast stroke analysis is not available."))

    # The analysed length is not known until the clip is opened, so the time
    # estimate starts from the requested range and is corrected once the real
    # frame count is in.
    if end_frame is not None:
        progress.frames = max(1, end_frame - start_frame + 1)

    progress.enter("reading")
    progress.enter("players")
    tracks = players.extract_players(
        video,
        start_frame=start_frame,
        end_frame=end_frame,
        on_frame=lambda done, total: progress.advance(done, total),
    )
    fps = tracks.fps
    frames = len(tracks.keypoints)
    progress.frames = frames

    # Opaque from here: TrackNetV3 is a subprocess that reports nothing this
    # module reads, so the stage is marked indeterminate and the interface shows
    # activity rather than a position nobody measured.
    progress.enter("shuttle")
    # The same range pose ran on, enforced inside TrackNetV3 rather than sliced
    # out of a whole-file result afterwards. The returned track is local to the
    # range: element i is source frame start_frame + i, exactly as
    # tracks.keypoints[i] is. Every source-frame number reported below is
    # therefore start_frame + a local index, and the two instruments are indexed
    # the same way.
    shuttle = shuttle_module.track_shuttle(video, frames, start_frame=start_frame)
    progress.enter("hits")
    if oracle_hits is None:
        hits = shuttle_module.detect_hits(shuttle, fps)
        hit_source = "detected"
    else:
        # Source frames -> range-local indices, clipped to the analysed range
        # and de-duplicated, so an annotation file covering a whole match can be
        # handed in whole. Sorted because the window builder reads neighbours.
        hits = sorted({
            int(f) - start_frame for f in oracle_hits
            if 0 <= int(f) - start_frame < frames
        })
        hit_source = "annotated"

    progress.enter("strokes")
    joints = np.stack([bst_adapter.normalize_joints(tracks.keypoints[i], tracks.boxes[i]) for i in range(frames)])
    joints[~tracks.both_present] = 0.0
    shuttle_norm = bst_adapter.normalize_shuttle(shuttle.xy, tracks.width, tracks.height)
    windows = bst_adapter.between_hits_windows(hits, fps, frames)

    classifier = bst_adapter.StrokeClassifier() if windows else None
    strokes: list[dict[str, Any]] = []
    for k, (start, hit, end) in enumerate(windows):
        progress.advance(k, len(windows))
        call = classifier.classify(joints[start:end + 1], shuttle_norm[start:end + 1])  # type: ignore[union-attr]
        # Whose hit, for the jump measurement. The classifier names a side; where
        # it does not (unknown), fall back to whichever wrist is nearer the shuttle.
        side_index = 0 if call.side == "top" else 1 if call.side == "bottom" else None
        if side_index is None and shuttle.visible[hit]:
            d = [
                np.linalg.norm(tracks.keypoints[hit, m, 10] - shuttle.xy[hit]) if tracks.keypoints[hit, m, 10].any() else np.inf
                for m in range(2)
            ]
            side_index = int(np.argmin(d)) if np.isfinite(min(d)) else None
        jump = (
            jump_evidence(tracks, hit, side_index, fps)
            if side_index is not None
            else {
                "indicator": "lowest-foot vertical pulse (image space)",
                "measures_physical_jump_height": False,
                "present": None,
                "evaluated": False,
                "reason": "the hitter could not be identified, so no pulse was evaluated",
                "note": FOOT_PULSE_STATEMENT,
                "rejected": JUMP_EVIDENCE_REJECTED,
                "rejected_on": JUMP_EVIDENCE_REJECTED_ON,
                "rejection": JUMP_EVIDENCE_REJECTION_STATEMENT,
            }
        )
        players_seen = float(tracks.both_present[start:end + 1].mean())
        shuttle_seen = float(shuttle.visible[start:end + 1].mean())
        # The gate, once. `state` keeps its exact previous meaning: it was
        # `fuse(call, jump)` overridden to "undetermined" when coverage failed,
        # and `not_determined_reason` is true in exactly the union of the
        # coverage failure and fuse's own undetermined branch.
        reason = not_determined_reason(
            call, both_players_seen=players_seen, shuttle_seen=shuttle_seen
        )
        state = "undetermined" if reason is not None else fuse(call, jump)
        stroke_type = (
            bst_adapter.NOT_DETERMINED if reason is not None else call.stroke
        )
        # The hitter is reported apart from the stroke, never folded into its
        # label. It comes from the same classification, so a call the gate
        # declined does not get to keep its side.
        hitter = call.side if reason is None else None
        strokes.append({
            **BOUNDARY,
            "index": k + 1,
            # The primary result. A smash is a smash whatever the foot-pulse
            # indicator did; `stroke_type` is computed without reading `jump`.
            "stroke_type": stroke_type,
            "stroke_type_label": STROKE_TYPE_LABELS.get(stroke_type, stroke_type),
            "stroke_type_group": STROKE_TYPE_GROUPS.get(stroke_type, "other"),
            "hitter": hitter,
            "hitter_label": (
                "Top player" if hitter == "top"
                else "Bottom player" if hitter == "bottom"
                else "Not determined"
            ),
            # Not a second prediction. BST_0's classes name side and stroke
            # jointly, and `call.side` is that label's prefix, so side accuracy
            # is a coarsening of stroke accuracy over one softmax rather than an
            # independent measurement of who hit the shuttle.
            "hitter_is_derived": True,
            "hitter_derivation": HITTER_SIDE_DERIVED_STATEMENT,
            "not_determined_reason": reason,
            # Source frame numbers alongside the seconds. The evaluation
            # protocol matches hits at +/-3 frames, and recovering a frame by
            # multiplying a 2-decimal second by the frame rate is an invitation
            # to an off-by-one in exactly the measurement that cannot afford it.
            "hit_frame": start_frame + hit,
            "window_start_frame": start_frame + start,
            "window_end_frame": start_frame + end,
            "hit_seconds": round((start_frame + hit) / fps, 2),
            "start_seconds": round((start_frame + start) / fps, 2),
            "end_seconds": round((start_frame + end) / fps, 2),
            "state": state,
            "label": STATE_LABELS[state],
            "state_means": STATE_MEANINGS[state],
            "stroke": bst_adapter.call_as_dict(call),
            "jump": jump,
            "coverage": {"both_players": round(players_seen, 3), "shuttle_visible": round(shuttle_seen, 3)},
            "replay": [],
        })

    if with_replays:
        progress.enter("replays")
        # The calls a reader most needs to check get their replay first, then
        # the rest in clip order until the cap. A cap that ran in clip order
        # dropped the replay of a likely jump smash at the end of a rally while
        # keeping fourteen frames of a serve.
        priority = {
            "likely_jump_smash": 0,
            "smash_no_detected_jump": 1,
            "smash_jump_not_evaluated": 1,
            "jump_overhead_other": 2,
        }
        order = sorted(range(len(strokes)), key=lambda i: (priority.get(strokes[i]["state"], 3), i))
        for rank, i in enumerate(order):
            progress.advance(rank, len(order))
            if rank < REPLAY_MAX_STROKES:
                start, _, end = windows[i]
                strokes[i]["replay"] = _replay(video, start_frame + start, start_frame + end)
            else:
                strokes[i]["replay_omitted"] = True
                strokes[i]["replay_omitted_reason"] = (
                    f"past the {REPLAY_MAX_STROKES}-stroke replay cap (ranked "
                    f"{rank + 1} of {len(order)}); the analysis of this stroke is "
                    "unaffected"
                )

    counts = {state: sum(1 for s in strokes if s["state"] == state) for state in STATES}
    # The stroke-type tally, kept apart from `counts` above rather than
    # replacing it: `counts` is the fusion-state tally every earlier payload
    # carried and is unchanged.
    type_counts = {
        stroke_type: sum(1 for s in strokes if s["stroke_type"] == stroke_type)
        for stroke_type in (*bst_adapter.STROKE_TYPES, bst_adapter.NOT_DETERMINED)
    }
    not_determined = type_counts[bst_adapter.NOT_DETERMINED]
    # The abstention rate is a headline number, not a footnote: a pipeline that
    # declines most of a rally is not a pipeline that got most of it right.
    abstained = counts["undetermined"] + counts["smash_jump_not_evaluated"]
    progress(1.0, "Done")

    # Computed once and used twice: the summary block, and the detector name
    # carried alongside the boxes so the interface can say where they came from.
    player_summary = players.as_summary(tracks)

    return {
        **BOUNDARY,
        "boundary": BOUNDARY_STATEMENT,
        "pose_substitution": MMPOSE_SUBSTITUTION_STATEMENT,
        "model_score_note": MODEL_SCORE_STATEMENT,
        "jump_evidence_note": FOOT_PULSE_STATEMENT,
        # The rejection, at the top of the payload rather than buried in each
        # stroke, so a reader who never opens a stroke card still meets it.
        "jump_evidence_rejected": {
            "rejected": JUMP_EVIDENCE_REJECTED,
            "rejected_on": JUMP_EVIDENCE_REJECTED_ON,
            "statement": JUMP_EVIDENCE_REJECTION_STATEMENT,
            "evidence": JUMP_EVIDENCE_REJECTION_EVIDENCE,
            "footage": DIAGNOSTIC_FOOTAGE,
            "fusion_states": FUSION_STATES_REJECTED_STATEMENT,
            "still_reported": (
                "The measurement is still computed and still carried in every stroke "
                "record. It is kept as the record of an unsuccessful experiment, not "
                "as a finding, and it is excluded from the formal evaluation."
            ),
        },
        "hitter_side_is_derived": HITTER_SIDE_DERIVED_STATEMENT,
        # Why some strokes have no replay. Stated as data so the interface does
        # not have to hard-code the cap it is explaining.
        "replay_policy": {
            "frames_per_stroke": REPLAY_FRAMES,
            "frame_width_px": REPLAY_WIDTH,
            "max_strokes": REPLAY_MAX_STROKES,
            "statement": (
                f"At most {REPLAY_MAX_STROKES} strokes carry replay frames. Every frame "
                "is a base64 JPEG inside this response, and embedding one for every hit "
                "in a long rally makes the response several times larger than the "
                "analysis it delivers."
            ),
            "priority": (
                "Strokes are ranked by fusion state before the cap is applied — the "
                "smash-group states first, then the rest in clip order — because a cap "
                "applied in clip order alone dropped the replay of a late-rally smash "
                "while keeping fourteen frames of an opening serve. A stroke past the "
                "cap carries replay_omitted: true and an empty replay list. Nothing "
                "about the analysis of that stroke differs; only its frames are absent."
            ),
        },
        "diagnostic_footage": {
            "footage": DIAGNOSTIC_FOOTAGE,
            "findings": DIAGNOSTIC_FINDINGS,
            "role": (
                "Development and diagnostic evidence only, spent in the same way the "
                "development rally is. Disqualified from the evaluation set and from "
                "every accuracy claim."
            ),
        },
        "rally_contract": {
            "statement": RALLY_CONTRACT_STATEMENT,
            "requirements": list(RALLY_CONTRACT),
            "enforced_by_the_server": "the analysed frame budget only",
            "not_full_match_analysis": True,
        },
        "video": {"frames": frames, "fps": round(fps, 3), "resolution": [tracks.width, tracks.height],
                  "start_frame": start_frame, "end_frame": start_frame + frames - 1,
                  "analysed_seconds": round(frames / fps, 2)},
        "stroke_types": {
            "primary_task": PRIMARY_TASK_STATEMENT,
            "per_class_performance": PER_CLASS_PERFORMANCE_STATEMENT,
            "taxonomy": list(bst_adapter.STROKE_TYPES),
            "labels": dict(STROKE_TYPE_LABELS),
            "groups": dict(STROKE_TYPE_GROUPS),
            "filters": [{"key": key, "label": label} for key, label in STROKE_FILTERS],
            "detected": len(strokes),
            "counts": type_counts,
            "not_determined": not_determined,
            "not_determined_rate": (
                round(not_determined / len(strokes), 3) if strokes else None
            ),
            "coverage_minimum": COVERAGE_MINIMUM,
            "score_note": MODEL_SCORE_STATEMENT,
            # Said here as well as in the fusion block, because this is the
            # block a reader of the primary result will stop at.
            "jump_evidence_is_supplementary": True,
            "jump_evidence_note": FOOT_PULSE_STATEMENT,
            "jump_evidence_rejected": JUMP_EVIDENCE_REJECTED,
            "hitter_side_is_derived": True,
            "hitter_side_note": HITTER_SIDE_DERIVED_STATEMENT,
        },
        "players": player_summary,
        # Where each player was, per analysed frame. Presentation data: the
        # dashboard's annotated rally replay draws these over the footage so a
        # stroke label can sit on the player it refers to instead of in the
        # middle of the picture. Nothing computes anything from them — no
        # threshold, no gate and no predicted class reads this block, and it
        # did not exist when any of them was frozen.
        #
        # Whole numbers, and small: two boxes per frame inside a 500-frame
        # budget is a few tens of kilobytes, against the base64 JPEGs the
        # replay frames already cost. There is no cap for the same reason.
        "player_boxes": {
            "space": "source-video pixels, [x1, y1, x2, y2]",
            "resolution": [tracks.width, tracks.height],
            "order": ["top", "bottom"],
            "start_frame": start_frame,
            "fps": round(fps, 3),
            "frame_count": frames,
            # A player who was not detected has an all-zero box. It must be
            # drawn as nothing: carrying the last box forward would put a
            # confident rectangle on a patch of court where the estimator saw
            # no one.
            "absent": (
                "An all-zero box means that player was not tracked in that frame. It "
                "is to be drawn as nothing, never as the last box that was."
            ),
            "detector": player_summary["pose_estimator"],
            "note": PLAYER_BOX_STATEMENT,
            "frames": np.rint(tracks.boxes[:frames]).astype(int).tolist(),
        },
        "shuttle": shuttle_module.as_summary(
            shuttle,
            hits,
            fps,
            start_frame=start_frame,
            hit_source=hit_source,
            # The space the trajectory is in, so the interface scales it by the
            # resolution the payload states rather than by the size it happens
            # to be rendered at.
            resolution=[tracks.width, tracks.height],
        ),
        "classifier": {
            "name": "BST_0 (Chang, CVPRW 2026) — ShuttleSet merged, 25 classes",
            "weight": bst_adapter.BST_WEIGHT.name,
            "score_is_calibrated": False,
            "score_note": MODEL_SCORE_STATEMENT,
            "stroke_model_score_threshold": STROKE_MODEL_SCORE_THRESHOLD,
            "pose_substitution": MMPOSE_SUBSTITUTION_STATEMENT,
        },
        "frozen_thresholds": {
            "frozen_on": FROZEN_ON,
            "informed_by": DEVELOPMENT_FOOTAGE,
            "statement": THRESHOLDS_STATEMENT,
            "values": FROZEN_THRESHOLDS,
        },
        "provenance": provenance(),
        "fusion_rule": {
            "likely_jump_smash": (
                f"the classifier's top class is smash with an uncalibrated model score "
                f">= {STROKE_MODEL_SCORE_THRESHOLD}, and the hitter's lowest foot rises "
                f">= {FOOT_PULSE_TORSOS} torso lengths in the image and returns, within "
                f"+/-{JUMP_WINDOW_SECONDS} s of the hit"
            ),
            "states": STATE_LABELS,
            "state_meanings": STATE_MEANINGS,
            # The rule is unchanged and still runs; what changed is that its
            # jump half has been rejected, so the states it produces are record
            # rather than result.
            "rejected": JUMP_EVIDENCE_REJECTED,
            "rejected_on": JUMP_EVIDENCE_REJECTED_ON,
            "rejection": FUSION_STATES_REJECTED_STATEMENT,
        },
        "stages": stage_plan(frames),
        "counts": counts,
        "abstained": abstained,
        "abstention_rate": round(abstained / len(strokes), 3) if strokes else None,
        "strokes": strokes,
    }
