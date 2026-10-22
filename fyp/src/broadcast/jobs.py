"""Broadcast analyses as background jobs, one at a time."""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.broadcast.frozen import RALLY_CONTRACT, RALLY_CONTRACT_STATEMENT
from src.broadcast import pipeline
from src.broadcast.run_clip import parse_progress_line

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "cache" / "broadcast_uploads"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
JOB_RETENTION_SECONDS = 60 * 60

MAX_ANALYSED_FRAMES = 500

MEASURED_SECONDS_PER_FRAME = 0.98
MEASURED_FIXED_SECONDS = 44.5


def estimated_runtime_seconds(frames: int) -> float:
    return MEASURED_FIXED_SECONDS + MEASURED_SECONDS_PER_FRAME * frames


def budget_seconds_at(fps: float) -> float:
    """The longest range, in seconds, that actually fits the budget at ``fps``.

    A range of N frames spans N-1 intervals, so 500 frames at 25 fps is 19.96 s
    of elapsed time, not 20.00 s. Rounded **down** to a tenth, because this
    number is published as the maximum a caller may ask for: advertising 20.0 s
    when 20.0 s is one frame over the budget recreates exactly the trap the
    frame budget was introduced to remove.
    """
    return math.floor((MAX_ANALYSED_FRAMES - 1) / (fps or 25.0) * 10) / 10


RUNTIME_NOTE = (
    f"Cost is per frame, not per second: about {MEASURED_SECONDS_PER_FRAME:.1f} s per "
    f"frame plus {MEASURED_FIXED_SECONDS:.0f} s of set-up, measured on the development "
    f"machine (Apple M3, CPU only, 720p). The {MAX_ANALYSED_FRAMES}-frame budget is "
    f"about {estimated_runtime_seconds(MAX_ANALYSED_FRAMES) / 60:.0f} minutes at any "
    f"frame rate - at most {budget_seconds_at(25)} s at 25 fps, "
    f"{budget_seconds_at(30)} s at 30 fps, {budget_seconds_at(60)} s at 60 fps."
)

MIN_PLAUSIBLE_FPS = 24.0
MAX_POSSIBLE_SECONDS = MAX_ANALYSED_FRAMES / MIN_PLAUSIBLE_FPS


def max_frames_for(fps: float) -> int:
    """The frame budget. Independent of ``fps`` - that is the point of it.

    Kept as a function taking ``fps`` so callers read the same way they did
    when the budget was a duration, and so the frame rate stays available if
    this ever needs to vary again.
    """
    return MAX_ANALYSED_FRAMES


CONTRACT = {
    "max_analysed_frames": MAX_ANALYSED_FRAMES,
    "equivalent_seconds": {
        "25_fps": budget_seconds_at(25),
        "30_fps": budget_seconds_at(30),
        "60_fps": budget_seconds_at(60),
    },
    "estimated_worst_case_minutes": round(estimated_runtime_seconds(MAX_ANALYSED_FRAMES) / 60, 1),
    "requirements": list(RALLY_CONTRACT),
    "statement": RALLY_CONTRACT_STATEMENT,
    "runtime_note": RUNTIME_NOTE,
    "not_full_match_analysis": True,
    "enforced_by_the_server": (
        "The analysed duration is enforced. The rest of the contract cannot be: nothing "
        "in the pipeline detects a replay, a close-up or a camera cut, so footage that "
        "breaks it is analysed as though it were a rally and the output is wrong "
        "without saying so."
    ),
}


class UploadRejected(ValueError):
    """The upload cannot be accepted at all."""


@dataclass
class BroadcastJob:
    job_id: str
    filename: str
    start_frame: int = 0
    end_frame: int | None = None
    status: str = "queued"
    progress: float = 0.0
    message: str | None = None
    stage: str | None = None
    stage_label: str | None = None
    stage_index: int | None = None
    stage_count: int | None = None
    stage_determinate: bool = True
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: dict[str, int] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "stage_index": self.stage_index,
            "stage_count": self.stage_count,
            "stage_determinate": self.stage_determinate,
            "stages": pipeline.stage_plan(),
            "longest_stage": pipeline.LONGEST_STAGE,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "result": self.result,
            "error": self.error,
        }


def upload_path(job_id: str, filename: str) -> Path:
    suffix = Path(filename).suffix.lower() or ".mp4"
    return UPLOAD_DIR / f"{job_id}{suffix}"


class BroadcastJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, BroadcastJob] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._threads_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        with self._threads_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_forever, name="broadcast-worker", daemon=True)
            self._worker.start()

    def submit(self, job: BroadcastJob) -> BroadcastJob:
        with self._lock:
            self._prune_locked()
            self._jobs[job.job_id] = job
            waiting = self._queue.qsize()
        if waiting:
            job.message = f"Waiting for {waiting} earlier clip{'s' if waiting > 1 else ''} to finish."
        self._queue.put(job.job_id)
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> BroadcastJob | None:
        with self._lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def _prune_locked(self) -> None:
        cutoff = time.time() - JOB_RETENTION_SECONDS
        for job_id in [j for j, job in self._jobs.items() if job.finished_at is not None and job.finished_at < cutoff]:
            job = self._jobs.pop(job_id, None)
            if job is not None:
                upload_path(job.job_id, job.filename).unlink(missing_ok=True)

    def _run_forever(self) -> None:  # pragma: no cover - thread body
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            path = upload_path(job.job_id, job.filename)
            try:
                job.status = "running"
                job.message = None

                def report(update: dict[str, Any]) -> None:
                    job.progress = min(0.99, float(update.get("progress", job.progress)))
                    job.message = update.get("message", job.message)
                    job.elapsed_seconds = float(update.get("elapsed_seconds", job.elapsed_seconds))
                    if "stage" in update:
                        job.stage = update["stage"]
                        job.stage_label = update.get("stage_label")
                        job.stage_index = update.get("stage_index")
                        job.stage_count = update.get("stage_count")
                        job.stage_determinate = bool(update.get("stage_determinate", True))
                    # Cleared rather than left stale: a remaining-time estimate
                    # that stops updating is worse than none at all.
                    job.estimated_remaining_seconds = update.get("estimated_remaining_seconds")

                job.result = analyse_in_subprocess(
                    path, start_frame=job.start_frame, end_frame=job.end_frame, report=report
                )
                job.status = "done"
                job.progress = 1.0
                job.message = None
                job.stage = None
                job.stage_label = None
                job.stage_determinate = True
                job.estimated_remaining_seconds = None
            except Exception as error:
                job.status = "failed"
                job.error = str(error)
            finally:
                job.finished_at = time.time()
                path.unlink(missing_ok=True)


SUBPROCESS_TIMEOUT_SECONDS = 3 * 3600


class AnalysisProcessFailed(RuntimeError):
    """The analysis process did not return a result."""


def _child_environment() -> dict[str, str]:
    """The child's environment: this one, plus an import path that reaches src.

    The child is started as ``-m src.broadcast.run_clip`` from the project root,
    so PYTHONPATH matters when the server was launched from somewhere else.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    root = str(PROJECT_ROOT)
    if root not in existing.split(os.pathsep):
        environment["PYTHONPATH"] = os.pathsep.join([root, existing]) if existing else root
    return environment


def _describe_exit(returncode: int) -> str:
    """Say what happened to the child, distinguishing a crash from a failure.

    A negative return code is a signal, and the signal is the whole diagnosis:
    ``SIGSEGV`` here means the analysis process died the way the API server used
    to die before this ran in a process of its own. Reporting it as "exit code
    -11" would bury exactly the fact worth knowing.
    """
    if returncode >= 0:
        return f"exited with code {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    return f"was killed by {name}"


def analyse_in_subprocess(
    video: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    report: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Analyse one clip in a fresh interpreter and return its result.

    The analysis cannot run in this process. ``src/broadcast/run_clip.py``
    documents why in full; the short version is that torch's OpenMP runtime
    alongside scikit-learn's and xgboost's segfaults the interpreter, and the
    API server is where those other two live. A separate process is not a
    tidiness measure here - it is the thing that keeps a failed analysis from
    being a failed server.

    Progress arrives as prefixed JSON lines on the child's stdout and is passed
    straight to ``report``, so a caller sees the same updates in the same shape
    it saw when this ran inline. Everything else the child's dependencies print
    is ignored rather than parsed.
    """
    with tempfile.TemporaryDirectory() as scratch:
        result_path = Path(scratch) / "result.json"
        request = {
            "video": str(Path(video).resolve()),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "result_path": str(result_path),
        }
        command = [sys.executable, "-m", "src.broadcast.run_clip", json.dumps(request)]
        child = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=_child_environment(),
            stdout=subprocess.PIPE,
            # Folded into stdout so the child's stderr cannot fill a pipe
            # nobody is draining and block a run that was otherwise fine.
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail: list[str] = []
        # A watchdog rather than ``wait(timeout=...)``: the loop below blocks on
        # the child's stdout until end of file, so a child that hangs while
        # holding the pipe open would never reach a wait() at all. Killing it
        # closes the pipe, which is what ends the read.
        timed_out = threading.Event()

        def _stop_child() -> None:  # pragma: no cover - fires only on a hang
            timed_out.set()
            child.kill()

        watchdog = threading.Timer(timeout, _stop_child)
        watchdog.daemon = True
        watchdog.start()
        try:
            assert child.stdout is not None
            for line in child.stdout:
                update = parse_progress_line(line)
                if update is not None:
                    if report is not None:
                        report(update)
                    continue
                # Kept for the failure message, bounded: a stalled child can
                # print without limit, and this is a diagnostic, not a log.
                line = line.rstrip()
                if line:
                    tail.append(line)
                    del tail[:-40]
            returncode = child.wait()
        finally:
            watchdog.cancel()
            if child.poll() is None:  # pragma: no cover - only on an unexpected exit path
                child.kill()
                child.wait()

        if timed_out.is_set():
            raise AnalysisProcessFailed(
                f"The analysis did not finish within {int(timeout // 60)} minutes and was stopped."
            )

        if result_path.exists():
            payload = json.loads(result_path.read_text())
            if "error" in payload:
                raise AnalysisProcessFailed(payload["error"])
            return payload

        # No result file and no error file: the child did not get far enough to
        # write either, which is the crash case this separation exists for.
        excerpt = "\n".join(tail[-8:])
        raise AnalysisProcessFailed(
            f"The analysis process {_describe_exit(returncode)} without producing a result."
            + (f" Its last output was:\n{excerpt}" if excerpt else "")
        )


STORE = BroadcastJobStore()


def sweep_orphaned_uploads() -> list[str]:
    """Delete every clip left in the broadcast upload directory. Startup only.

    The worker deletes an uploaded clip in a ``finally`` block, so a job that
    succeeds, fails normally, or dies inside TrackNetV3 all leave nothing
    behind. What that cannot survive is the process itself going away. The job
    store is in process memory, so a crash, a kill, or an ordinary API restart
    between writing the file and finishing with it strands the clip on disk
    with no job record left to explain it and nothing that will ever come back
    for it. Auditing this directory found 757 MB of broadcast footage stranded
    that way, some of it a 12,097-frame source recording.

    Deleting *everything* is safe here precisely because the store is in
    process memory: at startup there are no jobs, so nothing in this directory
    can belong to an analysis in flight. That reasoning depends on the
    single-worker deployment this module assumes, and it is why this must never
    be run periodically or from a second worker.

    Only files directly inside ``UPLOAD_DIR`` are touched — no recursion, no
    directories, and nothing anywhere else. A source recording a person
    selected a range from lives where they put it; this cache holds only the
    server's own copy of it.

    Returns the names removed, so startup can say what it did rather than
    deleting a person's footage silently.
    """
    if not UPLOAD_DIR.exists():
        return []
    removed: list[str] = []
    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            path.unlink()
        except OSError:
            # Report what was actually deleted. A file that could not be
            # removed would claim a deletion that did not happen.
            continue
        removed.append(path.name)
    return removed


def incoming_path() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR / f".incoming-{uuid.uuid4().hex}.part"


def adopt_upload(*, filename: str, source: Path, size: int, start_seconds: float | None, end_seconds: float | None) -> BroadcastJob:
    if size <= 0:
        raise UploadRejected("The uploaded file was empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"The file is {size / 1_073_741_824:.1f} GB and the upload limit is "
            f"{MAX_UPLOAD_BYTES // 1_073_741_824} GB. This limit is on the file, not on the "
            "analysis: trim the recording before uploading it, or upload a smaller export. "
            f"Only {MAX_ANALYSED_FRAMES} frames are analysed in any case, selected with the "
            "start and end times."
        )

    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if total <= 0:
        raise UploadRejected("The file could not be read as a video.")

    start = int(round((start_seconds or 0.0) * fps))
    end = total - 1 if end_seconds is None else int(round(end_seconds * fps))
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    budget = max_frames_for(fps)
    requested = end - start + 1
    if requested > budget:
        raise UploadRejected(
            f"The selected range is {requested} frames ({requested / fps:.1f} s at "
            f"{fps:.0f} fps), and this module analyses at most {budget} frames in one "
            f"run - {budget / fps:.1f} s at this clip's frame rate. Select one rally "
            "with the start and end times: a fixed rear-court view, no replay or "
            "close-up inside the range, both players and the shuttle visible throughout. "
            "This is not full-match analysis. " + RUNTIME_NOTE
        )

    job = BroadcastJob(job_id=uuid.uuid4().hex, filename=Path(filename).name or "clip.mp4", start_frame=start, end_frame=end)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    source.replace(upload_path(job.job_id, job.filename))
    return STORE.submit(job)
