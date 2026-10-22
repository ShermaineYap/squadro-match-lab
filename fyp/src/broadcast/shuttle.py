"""The shuttlecock's trajectory, and the hits read off it."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.broadcast.frozen import (
    MIN_HIT_SEPARATION_SECONDS,
    SHUTTLE_REVERSAL_PX,
    SHUTTLE_SMOOTHING_FRAMES,
    SHUTTLE_TRACK_STATEMENT,
)
from src.broadcast.third_party import INPAINTNET_WEIGHT, TRACKNET_REPO, TRACKNET_WEIGHT

# The three numbers that decide what counts as a hit are imported from
# src.broadcast.frozen rather than defined here, so there is one frozen record
# of every exploratory threshold in the module instead of several copies that
# can drift apart. They are development choices, not fitted values; see
# frozen.THRESHOLDS_STATEMENT.
__all__ = [
    "MIN_HIT_SEPARATION_SECONDS",
    "SHUTTLE_REVERSAL_PX",
    "SHUTTLE_SMOOTHING_FRAMES",
    "ClipVanished",
    "ShuttleTrack",
    "TrackNetFailed",
    "as_summary",
    "detect_hits",
    "track_shuttle",
]

TRACKNET_NUM_WORKERS = 0

TRACKNET_ERROR_HEAD = 2000
TRACKNET_ERROR_TAIL = 6000


@dataclass
class ShuttleTrack:
    xy: np.ndarray  # (t, 2) pixels
    visible: np.ndarray  # (t,) bool

    @property
    def visible_fraction(self) -> float:
        return float(self.visible.mean()) if len(self.visible) else 0.0


def _mp4_named(video: Path, scratch: Path) -> Path:
    """The same clip under a name TrackNetV3's assertion will accept.

    ``utils/general.py`` guards its frame reader with

        assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    which is a check on the *filename*, not on the container: two lines later
    the file goes to ``cv2.VideoCapture``, which decodes by content and does not
    look at the extension at all. So a QuickTime ``.mov`` or an ``.m4v`` that
    OpenCV reads perfectly well is refused for how it is spelled.

    That refusal used to land at the shuttle-tracking stage, several minutes
    into a job, after pose had already run over every frame. The clip's
    readability is established long before that: ``jobs.adopt_upload`` opens it
    with OpenCV to read its frame rate and length and rejects anything OpenCV
    cannot decode. Anything reaching here is therefore already known-good, and
    the only thing standing in the way is the spelling.

    A symlink is enough - OpenCV follows it and no bytes are copied. Where the
    filesystem refuses one, fall back to a copy in the same scratch directory,
    which is removed with everything else when the analysis ends.
    """
    if video.suffix.lower() == ".mp4":
        return video
    alias = scratch / f"{video.stem}.mp4"
    try:
        alias.symlink_to(video.resolve())
    except OSError:  # noqa: BLE001 - no symlink support here; spend the copy
        shutil.copy2(video, alias)
    return alias


class ClipVanished(RuntimeError):
    """The clip was there when the analysis started and is not there now.

    Kept apart from ``TrackNetFailed`` because it is not a failure of
    TrackNetV3: the tracker never ran. Reporting it as an exit code would name
    the wrong component and send someone reading the message into third-party
    code that behaved correctly.
    """


class TrackNetFailed(RuntimeError):
    """The TrackNetV3 subprocess exited non-zero.

    Carries the return code and the excerpt separately as well as in the
    message, so a caller can report the cause rather than re-parse a string.
    """

    def __init__(self, returncode: int, command: list[str], excerpt: str) -> None:
        self.returncode = returncode
        self.command = list(command)
        self.excerpt = excerpt
        super().__init__(
            f"TrackNetV3 exited with code {returncode}.\n"
            f"Command: {' '.join(command)}\n{excerpt}"
        )


_PROGRESS_BAR = re.compile(r"^\s*\d+%\|.*\|\s*\d+/\d+\s*\[")


def _without_progress_redraws(stream: str) -> list[str]:
    """Every line of ``stream``, with tqdm's redraws collapsed to the last one.

    tqdm redraws its bar with a carriage return, and ``str.splitlines`` splits
    on a carriage return, so one progress bar arrives here as hundreds of lines
    that are the same bar at different percentages. Keeping the last of each
    consecutive run keeps the only information the bar carries — how far the
    run got before it died — and stops it from pushing the traceback out of a
    bounded excerpt. Nothing that is not a progress bar is dropped.
    """
    lines: list[str] = []
    run_of_bars = 0
    for line in stream.splitlines():
        if _PROGRESS_BAR.match(line):
            if run_of_bars:
                lines[-1] = line
            else:
                lines.append(line)
            run_of_bars += 1
        else:
            run_of_bars = 0
            lines.append(line)
    return lines


def _bounded(text: str) -> str:
    """``text``, or its head and tail with the omission stated in the middle."""
    budget = TRACKNET_ERROR_HEAD + TRACKNET_ERROR_TAIL
    if len(text) <= budget:
        return text
    omitted = len(text) - budget
    return (
        text[:TRACKNET_ERROR_HEAD]
        + f"\n… {omitted} characters omitted from the middle of this stream …\n"
        + text[-TRACKNET_ERROR_TAIL:]
    )


def failure_excerpt(completed: subprocess.CompletedProcess) -> str:
    """Both streams of a failed run, bounded, with neither one discarded.

    The handling this replaces kept the last three lines of one stream. On a
    CPU run those three lines are always tqdm's final bar and the resource
    tracker's semaphore complaint, which say nothing about why the run failed —
    the traceback that does is a screen further up, and was thrown away.
    """
    parts: list[str] = []
    for name, stream in (("stderr", completed.stderr), ("stdout", completed.stdout)):
        text = "\n".join(_without_progress_redraws(stream or "")).strip()
        if text:
            parts.append(f"--- {name} ---\n{_bounded(text)}")
    return "\n".join(parts) if parts else "(the process wrote nothing to stdout or stderr)"


def track_shuttle(
    video: Path,
    frames: int,
    *,
    start_frame: int = 0,
    eval_mode: str = "nonoverlap",
    num_workers: int = TRACKNET_NUM_WORKERS,
) -> ShuttleTrack:
    """Run TrackNetV3 over ``frames`` frames from ``start_frame`` and return
    per-frame shuttle positions **local to that range**.

    ``start_frame`` is an index into the *source file*; the returned arrays are
    indexed from zero at ``start_frame``, so element ``i`` is source frame
    ``start_frame + i``. The caller holds the offset — it is the same offset
    ``players.extract_players`` was given — and nothing here needs slicing
    afterwards.

    That mapping used to be a lie told after the fact. TrackNetV3 was handed
    the file with no range at all, read it to the end, and inferred on every
    frame of it; the caller then sliced the requested window out of a whole-file
    result. On the development recording that meant 12,097 frames tracked for a
    190-frame selection — 190 batches instead of three, hours instead of
    minutes, and a background median taken over a whole match rather than over
    the rally. ``patches/tracknetv3-frame-range.patch`` gives the reader and
    the command line a range so the selection is enforced where the inference
    happens.

    ``nonoverlap`` runs each frame through the network once. The upstream
    default, ``weight``, ensembles overlapping windows. Measured on one 16 s
    broadcast rally on this machine: 200 s against roughly 20 minutes, identical
    visibility, a 2 px median position difference — and a 171 px 90th
    percentile, so a minority of frames do diverge. The hit list, which is what
    the classifier windows are cut on, came out the same from both apart from
    one extra reversal at the clip's end. Not a trade worth reversing on a CPU.
    """
    if frames <= 0:
        raise ValueError("track_shuttle needs at least one frame.")
    if start_frame < 0:
        raise ValueError("track_shuttle needs a non-negative start frame.")
    end_frame = start_frame + frames - 1
    # The subprocess runs in the TrackNetV3 checkout, so a relative path here
    # would be resolved against *its* directory. Callers in this project pass
    # absolute paths; this makes the ones that do not fail early and obviously
    # rather than inside a third-party traceback.
    video = video.resolve()
    # Checked here because this stage runs minutes after the clip was read for
    # the first time, and the clip can go away in between: the upload cache is
    # swept wholesale by a starting server, so a second server launched during
    # an analysis used to delete this file. TrackNetV3's own report of that is
    # unreadable - OpenCV cannot open the path, its frame reader returns an
    # empty list, and the failure surfaces as ``IndexError: too many indices for
    # array`` from indexing a 1-dimensional array four ways, several frames
    # deep in third-party code. Say what actually happened instead.
    if not video.exists():
        raise ClipVanished(
            f"The uploaded clip is no longer on disk: {video}. It was read at the "
            "start of this analysis, so it was deleted while the analysis was "
            "running - most likely by a second API server process, which empties "
            "the upload cache as it starts up. Run one server at a time, then "
            "upload the clip again."
        )
    with tempfile.TemporaryDirectory() as scratch:
        source = _mp4_named(video, Path(scratch))
        command = [
            sys.executable, "predict.py",
            "--video_file", str(source),
            "--tracknet_file", str(TRACKNET_WEIGHT),
            "--inpaintnet_file", str(INPAINTNET_WEIGHT),
            "--save_dir", scratch,
            "--batch_size", "8",
            "--eval_mode", eval_mode,
            # The range is not advisory. Without these two the tracker reads
            # the whole uploaded recording; see the patch note above.
            "--start_frame", str(start_frame),
            "--end_frame", str(end_frame),
            "--num_workers", str(num_workers),
        ]
        completed = subprocess.run(
            command, cwd=str(TRACKNET_REPO), capture_output=True, text=True, timeout=3600
        )
        if completed.returncode != 0:
            raise TrackNetFailed(completed.returncode, command, failure_excerpt(completed))
        csv_path = Path(scratch) / f"{source.stem}_ball.csv"
        if not csv_path.exists():
            raise RuntimeError("TrackNetV3 produced no trajectory file.")
        import pandas as pd

        table = pd.read_csv(csv_path)

    # TrackNet numbers the frames it was given from zero, and it was given the
    # selected range, so its indices are already this track's indices. The
    # bound is a guard against the padded final sequence, which can name a
    # frame past the end of the range.
    xy = np.zeros((frames, 2))
    visible = np.zeros(frames, dtype=bool)
    for row in table.itertuples():
        index = int(row.Frame)
        if 0 <= index < frames and int(row.Visibility) == 1:
            xy[index] = (float(row.X), float(row.Y))
            visible[index] = True
    return ShuttleTrack(xy, visible)


def detect_hits(track: ShuttleTrack, fps: float) -> list[int]:
    """Frame indices where the shuttle reversed vertical direction."""
    import pandas as pd

    if len(track.xy) < 5:
        return []
    y = pd.Series(np.where(track.visible, track.xy[:, 1], np.nan))
    y = y.interpolate(limit_direction="both").rolling(SHUTTLE_SMOOTHING_FRAMES, center=True, min_periods=1).mean()
    y = y.to_numpy()
    if np.isnan(y).all():
        return []
    velocity = np.gradient(np.nan_to_num(y))
    separation = int(fps * MIN_HIT_SEPARATION_SECONDS)
    hits: list[int] = []
    for i in range(2, len(y) - 2):
        before, after = velocity[i - 1], velocity[i + 1]
        if np.sign(before) != np.sign(after) and abs(before - after) > SHUTTLE_REVERSAL_PX:
            if not hits or i - hits[-1] >= separation:
                hits.append(i)
    return hits


def as_summary(
    track: ShuttleTrack,
    hits: list[int],
    fps: float,
    *,
    start_frame: int = 0,
    hit_source: str = "detected",
    resolution: list[int] | None = None,
) -> dict[str, Any]:
    """The trajectory summary, with every moment given in **source** terms.

    ``hits`` are indices into the track, which is local to the selected range.
    Reporting them as ``h / fps`` gave seconds counted from the start of the
    *range*, while every stroke in the same payload reported seconds counted
    from the start of the *file* — so one hit appeared twice, 200 s apart,
    under the same name. ``start_frame`` is applied here for the same reason it
    is applied to each stroke: a payload should have one clock.
    """
    return {
        "tracker": "TrackNetV3 with rectification module (qaz812345)",
        "visible_fraction": round(track.visible_fraction, 3),
        # The trajectory itself, for the dashboard's annotated replay to draw.
        # A copy for presentation: the hits below are read from the array this
        # was copied from, and nothing reads this copy back.
        "track": {
            "space": "source-video pixels, [x, y] of the tracked shuttle",
            "resolution": list(resolution) if resolution else None,
            "start_frame": start_frame,
            "fps": round(fps, 3),
            "frame_count": int(len(track.visible)),
            "absent": (
                "null means the tracker did not see the shuttle in that frame. It is "
                "to be drawn as nothing, never interpolated between the points either "
                "side of it."
            ),
            "note": SHUTTLE_TRACK_STATEMENT,
            "positions": [
                [int(round(float(x))), int(round(float(y)))] if seen else None
                for (x, y), seen in zip(track.xy, track.visible)
            ],
        },
        "hits": len(hits),
        "hit_frames": [start_frame + h for h in hits],
        "hit_seconds": [round((start_frame + h) / fps, 2) for h in hits],
        "moments_are": "source frames and source time, counted from the start of the uploaded file",
        # Which instrument put the hits there. Always "detected" on the API and
        # the dashboard; "annotated" only when the evaluation harness supplied
        # manual contact frames, so a results table can never confuse an
        # oracle-hit run with an end-to-end one.
        "hit_source": hit_source,
        "hit_source_means": (
            "detected: hits are reversals of the tracked shuttle's vertical velocity, the "
            "only path the API and the dashboard use. annotated: hits were supplied from "
            "manual ground-truth contact frames by the evaluation harness, to measure the "
            "classifier without hit-detection error in front of it."
        ),
        "hit_rule": (
            "A hit is a reversal of the shuttle's smoothed vertical velocity larger than "
            f"{SHUTTLE_REVERSAL_PX} px/frame, at least {MIN_HIT_SEPARATION_SECONDS} s from "
            "the previous one. These are frozen exploratory thresholds, not fitted values."
        ) if hit_source == "detected" else (
            "Hits were not detected for this run: they were supplied from manual "
            "annotation. The frozen reversal thresholds were not applied."
        ),
    }
