"""Run one clip's analysis in a process of its own. Child entry point."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROGRESS_PREFIX = "@@broadcast-progress@@ "


def emit_progress(update: dict[str, Any]) -> None:
    """Write one progress update for the parent, and flush it.

    Unbuffered on purpose: this is the only evidence the parent has that a run
    which takes minutes is still moving, so it is worth a syscall per stage.
    """
    sys.stdout.write(PROGRESS_PREFIX + json.dumps(update) + "\n")
    sys.stdout.flush()


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """Read one line of the child's stdout; ``None`` if it is not ours.

    Everything the child's dependencies print shares this stream, so "not a
    progress update" is the ordinary case and has to be cheap and silent.
    A prefixed line that will not parse is also ``None``: a mangled update -
    interleaved with another writer's output, most likely - is worth dropping,
    never worth failing a run that is otherwise going fine.
    """
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        update = json.loads(line[len(PROGRESS_PREFIX):])
    except ValueError:
        return None
    return update if isinstance(update, dict) else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m src.broadcast.run_clip '<request json>'", file=sys.stderr)
        return 2
    request = json.loads(argv[1])
    result_path = Path(request["result_path"])

    # Imported here rather than at module scope so that a malformed request is
    # answered in milliseconds instead of after torch has finished loading.
    from src.broadcast.pipeline import Progress, analyse_broadcast_clip

    try:
        result = analyse_broadcast_clip(
            Path(request["video"]),
            start_frame=int(request.get("start_frame", 0)),
            end_frame=request.get("end_frame"),
            progress=Progress(emit_progress, frames=int(request.get("frames", 0))),
        )
    except Exception as error:
        # Written, not raised: the parent has no traceback of its own for a
        # process it only watched, and "it failed" is not a report.
        result_path.write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        )
        return 1

    result_path.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
