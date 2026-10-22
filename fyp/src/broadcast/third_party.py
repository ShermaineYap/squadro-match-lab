"""Where the third-party code and weights live, whether they are present, and whether
they are the *same* code and weights the module was developed against.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = PROJECT_ROOT / "third_party"
WEIGHTS_DIR = PROJECT_ROOT / "models" / "broadcast"

BST_REPO = THIRD_PARTY / "BST-Badminton-Stroke-type-Transformer"
BST_CODE = BST_REPO / "stroke_classification"
TRACKNET_REPO = THIRD_PARTY / "TrackNetV3"

BST_WEIGHT = WEIGHTS_DIR / "bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_9.pt"
BST_WEIGHT_GDRIVE_ID = "1QH_bR-eXDAfV13jWEBcvUyBuyO87jDbZ"  # the same file, in the author's shared folder
TRACKNET_WEIGHT = WEIGHTS_DIR / "TrackNet_best.pt"
INPAINTNET_WEIGHT = WEIGHTS_DIR / "InpaintNet_best.pt"
TRACKNET_CKPTS_GDRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
YOLO_POSE_WEIGHT = WEIGHTS_DIR / "yolo11m-pose.pt"

BST_COMMIT = "fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df"
TRACKNET_COMMIT = "77c123ad4dd449b7d275f16cc43f316ba5b54042"

WEIGHT_SHA256: dict[str, str] = {
    "bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_9.pt":
        "db1f319f0a3f3343d7cfaeac0b32fbefc6d845f9a97ec216865136c91e950cd8",
    "TrackNet_best.pt":
        "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a",
    "InpaintNet_best.pt":
        "5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c",
    "yolo11m-pose.pt":
        "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2",
}

TRACKNET_CPU_PATCH = PROJECT_ROOT / "patches" / "tracknetv3-cpu.patch"
TRACKNET_FRAME_RANGE_PATCH = PROJECT_ROOT / "patches" / "tracknetv3-frame-range.patch"

TRACKNET_PATCHES: tuple[Path, ...] = (TRACKNET_CPU_PATCH, TRACKNET_FRAME_RANGE_PATCH)

PATCH_SHA256: dict[str, str] = {
    "tracknetv3-cpu.patch":
        "43209e6d47a6c47fee801fec31bcb5a7a2ad9d53352e6885c5675c44bc662864",
    "tracknetv3-frame-range.patch":
        "7b836826f4917ec0992b50602424858794d7ecfd44e373991a959372e7a1ad5f",
}


class ProvenanceError(RuntimeError):
    """A pinned commit or a recorded weight hash did not match."""


def sha256_of(path: Path, *, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def head_commit(repo: Path) -> str | None:
    """The checked-out commit of a clone, or ``None`` if it is not one."""
    import subprocess

    if not (repo / ".git").exists():
        return None
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except Exception:  # noqa: BLE001 - no git, or a broken clone
        return None
    return done.stdout.strip() or None


def verify_commits() -> list[str]:
    """Problems with the checked-out upstream commits; empty when both match."""
    problems: list[str] = []
    for name, repo, pinned in (
        ("BST", BST_REPO, BST_COMMIT),
        ("TrackNetV3", TRACKNET_REPO, TRACKNET_COMMIT),
    ):
        found = head_commit(repo)
        if found is None:
            problems.append(f"{name}: not cloned at {repo}, so its commit cannot be verified.")
        elif found != pinned:
            problems.append(f"{name}: checked out {found}, pinned to {pinned}.")
    return problems


def verify_weights() -> list[str]:
    """Problems with the weight files; empty when all four match their hash."""
    problems: list[str] = []
    for name, expected in WEIGHT_SHA256.items():
        path = WEIGHTS_DIR / name
        if not path.exists():
            problems.append(f"{name}: missing from {WEIGHTS_DIR}.")
            continue
        found = sha256_of(path)
        if found != expected:
            problems.append(f"{name}: SHA-256 {found}, recorded {expected}.")
    return problems


def patch_state(patch: Path) -> str:
    """Whether ``patch`` is applied to the TrackNetV3 checkout."""
    import subprocess

    if not (TRACKNET_REPO / ".git").exists() or not patch.exists():
        return "unknown"
    applied = subprocess.run(
        ["git", "-C", str(TRACKNET_REPO), "apply", "--reverse", "--check", str(patch)],
        capture_output=True, text=True,
    )
    if applied.returncode == 0:
        return "applied"
    forward = subprocess.run(
        ["git", "-C", str(TRACKNET_REPO), "apply", "--check", str(patch)],
        capture_output=True, text=True,
    )
    return "not applied" if forward.returncode == 0 else "does not apply"


def verify_patches() -> list[str]:
    """Problems with the patch files and their application; empty when clean.

    Two separate questions, both of which can fail silently otherwise: is the
    patch file the recorded one, and is it actually applied to the checkout on
    disk. A checkout that lost the frame-range patch would go back to tracking
    the whole uploaded recording and would say nothing about it.
    """
    problems: list[str] = []
    for patch in TRACKNET_PATCHES:
        if not patch.exists():
            problems.append(f"{patch.name}: missing from {patch.parent}.")
            continue
        expected = PATCH_SHA256.get(patch.name)
        found = sha256_of(patch)
        if expected is not None and found != expected:
            problems.append(f"{patch.name}: SHA-256 {found}, recorded {expected}.")
        state = patch_state(patch)
        if state != "applied":
            problems.append(f"{patch.name}: {state} to the TrackNetV3 checkout.")
    return problems


def provenance() -> dict[str, Any]:
    """Pins, recorded hashes and what is actually on disk, for the payload."""
    commit_problems = verify_commits()
    weight_problems = verify_weights()
    patch_problems = verify_patches()
    return {
        "pinned_commits": {"bst": BST_COMMIT, "tracknet_v3": TRACKNET_COMMIT},
        "checked_out_commits": {
            "bst": head_commit(BST_REPO),
            "tracknet_v3": head_commit(TRACKNET_REPO),
        },
        "recorded_weight_sha256": dict(WEIGHT_SHA256),
        "recorded_patch_sha256": dict(PATCH_SHA256),
        "tracknet_patches": {
            patch.name: {
                "recorded_sha256": PATCH_SHA256.get(patch.name),
                "found_sha256": sha256_of(patch) if patch.exists() else None,
                "state": patch_state(patch),
            }
            for patch in TRACKNET_PATCHES
        },
        "commits_verified": not commit_problems,
        "weights_verified": not weight_problems,
        "patches_verified": not patch_problems,
        "problems": commit_problems + weight_problems + patch_problems,
    }

ENABLE_ENV = "ENABLE_BROADCAST_STROKE_ANALYSIS"
_DISABLED_VALUES = frozenset({"off", "0", "false", "no"})


def enabled(environment: dict[str, str] | None = None) -> bool:
    source = os.environ if environment is None else environment
    return source.get(ENABLE_ENV, "on").strip().lower() not in _DISABLED_VALUES


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:  # ImportError, but also a torch that fails to load
        return False


def add_bst_to_path() -> None:
    """Make ``model.bst`` and ``preparing_data.*`` importable from the clone."""
    code = str(BST_CODE)
    if code not in sys.path:
        sys.path.insert(0, code)


def dependency_status() -> dict[str, Any]:
    """What this module needs, and what is actually present."""
    status: dict[str, Any] = {"enabled": enabled()}
    for name, module in (
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("opencv", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("positional_encodings", "positional_encodings"),
    ):
        status[name] = _importable(module)
    status["bst_code"] = (BST_CODE / "model" / "bst.py").exists()
    status["tracknet_code"] = (TRACKNET_REPO / "predict.py").exists()
    status["bst_weight"] = BST_WEIGHT.exists()
    status["tracknet_weight"] = TRACKNET_WEIGHT.exists() and INPAINTNET_WEIGHT.exists()
    status["yolo_pose_weight"] = YOLO_POSE_WEIGHT.exists()
    status["cpu_patch_recorded"] = TRACKNET_CPU_PATCH.exists()
    status["frame_range_patch_recorded"] = TRACKNET_FRAME_RANGE_PATCH.exists()
    # A checkout missing the frame-range patch does not fail — it quietly
    # tracks the whole uploaded recording and returns a trajectory for frames
    # nobody asked about. That is exactly the defect this is here to catch, so
    # it is part of "ready" rather than a note beside it.
    status["frame_range_patch_applied"] = patch_state(TRACKNET_FRAME_RANGE_PATCH) == "applied"
    status["ready"] = status["enabled"] and all(
        status[key]
        for key in (
            "torch", "ultralytics", "opencv", "numpy", "pandas", "positional_encodings",
            "bst_code", "tracknet_code", "bst_weight", "tracknet_weight",
            "frame_range_patch_applied",
        )
    )
    if not status["enabled"]:
        status["install_hint"] = f"Set {ENABLE_ENV}=on to expose this experimental module."
    elif not status["ready"]:
        status["install_hint"] = (
            "pip install -r requirements-broadcast.txt && "
            "python3 scripts/setup_broadcast_module.py"
        )
        if not status["frame_range_patch_applied"] and status["tracknet_code"]:
            status["install_hint"] = (
                "The TrackNetV3 checkout is missing patches/tracknetv3-frame-range.patch, "
                "without which the tracker reads the whole uploaded recording rather than "
                "the selected range. Re-run: python3 scripts/setup_broadcast_module.py"
            )
    return status
