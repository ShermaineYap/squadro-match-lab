"""Feeding BST_0 exactly what it was trained on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.broadcast.frozen import MERGED_STROKE_CLASSES, MODEL_SCORE_STATEMENT
from src.broadcast.third_party import BST_WEIGHT, add_bst_to_path

SEQ_LEN = 100
N_JOINTS = 17
N_CLASSES = 25

STROKE_NAMES: dict[str, str] = {
    "未知球種": "unknown",
    "放小球": "net shot",
    "擋小球": "block",
    "殺球": "smash",
    "挑球": "lift",
    "長球": "clear",
    "平球": "drive",
    "切球": "drop",
    "推球": "push",
    "撲球": "net kill",
    "勾球": "cross-court net",
    "發短球": "short serve",
    "發長球": "long serve",
}

OVERHEAD_STROKES = frozenset({"smash", "clear", "drop"})

STROKE_TYPES: tuple[str, ...] = (
    "smash",
    "drop",
    "clear",
    "lift",
    "drive",
    "net shot",
    "block",
    "push",
    "net kill",
    "cross-court net",
    "short serve",
    "long serve",
)

NOT_DETERMINED = "not determined"


@dataclass(frozen=True)
class StrokeCall:
    """One window's call.

    ``model_score`` and the scores in ``top3`` are **uncalibrated model
    scores** - raw softmax outputs from a network trained on other footage,
    with another pose estimator, and never calibrated on anything this project
    has. They are named ``score`` and not ``probability`` or ``confidence``
    throughout, because a number called a probability will be read as one.
    """

    stroke: str  # English stroke name
    side: str | None  # "top" | "bottom" | None for unknown
    model_score: float
    top3: tuple[tuple[str, str | None, float], ...]
    smash_score_top: float
    smash_score_bottom: float


def normalize_joints(arr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """``prepare_train_on_shuttleset.normalize_joints`` with center_align=True.

    ``arr`` (m, J, 2) pixels, ``bbox`` (m, 4) xyxy. Missing joints are zero in
    and zero out, which is how the upstream masks them.
    """
    dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    dist = np.where(dist == 0, 1.0, dist)
    ax, ay = arr[:, :, 0], arr[:, :, 1]
    xn = np.where(ax != 0.0, (ax - bbox[:, None, 0]) / dist, 0.0)
    yn = np.where(ay != 0.0, (ay - bbox[:, None, 1]) / dist, 0.0)
    centre = (bbox[:, :2] + bbox[:, 2:]) / 2
    cn = (centre - bbox[:, :2]) / dist
    xn = np.where(ax != 0.0, xn - cn[:, None, 0], 0.0)
    yn = np.where(ay != 0.0, yn - cn[:, None, 1], 0.0)
    return np.stack((xn, yn), axis=-1)


def normalize_shuttle(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """``normalize_shuttlecock``: divided by the video resolution."""
    return np.stack((xy[:, 0] / width, xy[:, 1] / height), axis=-1)


def between_hits_windows(hits: Sequence[int], fps: float, frames: int) -> list[tuple[int, int, int]]:
    """``gen_my_dataset`` strategy ``between_2_hits_with_max_limits``.

    Each window runs from the previous hit to the next hit plus 0.25 s, clamped
    to 1.5 s either side of its own hit. Returns (start, hit, end) per hit.
    """
    half = int(fps // 2)
    limit = int(fps * 3 // 2)
    eps = half // 2
    windows: list[tuple[int, int, int]] = []
    for k, hit in enumerate(hits):
        start = hits[k - 1] if k > 0 else hit - half
        end = (hits[k + 1] + eps) if k + 1 < len(hits) else hit + half
        start = max(start, hit - limit, 0)
        end = min(end, hit + limit + eps, frames - 1)
        if end > start:
            windows.append((start, hit, end))
    return windows


class StrokeClassifier:
    """BST_0 with the ShuttleSet-merged weights, on CPU."""

    def __init__(self) -> None:
        import torch

        add_bst_to_path()
        from model.bst import BST_0  # type: ignore[import-not-found]
        from preparing_data.shuttleset_dataset import (  # type: ignore[import-not-found]
            create_bones,
            get_bone_pairs,
            get_merged_stroke_types,
            make_seq_len_same,
        )

        self._torch = torch
        self._create_bones = create_bones
        self._make_seq_len_same = make_seq_len_same
        self._pairs = get_bone_pairs()
        self._classes = get_merged_stroke_types()
        # Every label this module prints is an index into this list. If the
        # clone's order is not the order recorded when the weights were chosen,
        # the outputs are silently relabelled rather than wrong-looking.
        if tuple(self._classes) != MERGED_STROKE_CLASSES:
            raise RuntimeError(
                "The BST clone's merged class list does not match the 25-class mapping "
                "recorded in src/broadcast/frozen.py. Re-run "
                "scripts/setup_broadcast_module.py to restore the pinned commit."
            )
        net = BST_0(
            in_dim=(N_JOINTS + len(self._pairs)) * 2,
            n_class=N_CLASSES,
            seq_len=SEQ_LEN,
            depth_tem=2,
            depth_inter=1,
        )
        net.load_state_dict(torch.load(str(BST_WEIGHT), map_location="cpu", weights_only=True))
        net.eval()
        self._net = net

    @staticmethod
    def _split(label: str) -> tuple[str, str | None]:
        side = "top" if label.startswith("Top_") else "bottom" if label.startswith("Bottom_") else None
        core = label.replace("Top_", "").replace("Bottom_", "")
        return STROKE_NAMES.get(core, core), side

    def classify(
        self,
        joints: np.ndarray,  # (t, 2, 17, 2) already normalised, zero where a player is absent
        shuttle: np.ndarray,  # (t, 2) normalised
    ) -> StrokeCall:
        torch = self._torch
        bones = self._create_bones(joints, self._pairs)
        JnB = np.concatenate((joints, bones), axis=-2)
        positions = np.zeros((len(joints), 2, 2))
        JnB, _, shuttle, length = self._make_seq_len_same(SEQ_LEN, JnB, positions, shuttle)
        x = torch.tensor(JnB, dtype=torch.float32)[None].reshape(1, SEQ_LEN, 2, -1)
        with torch.no_grad():
            logits = self._net(
                x, torch.tensor(shuttle, dtype=torch.float32)[None], torch.tensor([length])
            )
        # Softmax over the logits. This is an uncalibrated model score, not a
        # probability that the call is correct, and is named so everywhere it
        # travels.
        scores = torch.softmax(logits, dim=1)[0].numpy()
        order = np.argsort(scores)[::-1]
        top3 = tuple(
            (*self._split(self._classes[i]), round(float(scores[i]), 4)) for i in order[:3]
        )
        stroke, side = self._split(self._classes[order[0]])
        smash_top = float(scores[self._classes.index("Top_殺球")])
        smash_bottom = float(scores[self._classes.index("Bottom_殺球")])
        return StrokeCall(
            stroke=stroke,
            side=side,
            model_score=float(scores[order[0]]),
            top3=top3,
            smash_score_top=smash_top,
            smash_score_bottom=smash_bottom,
        )


def call_as_dict(call: StrokeCall) -> dict[str, Any]:
    return {
        "stroke": call.stroke,
        "side": call.side,
        "model_score": round(call.model_score, 4),
        "top3": [{"stroke": s, "side": d, "model_score": p} for s, d, p in call.top3],
        "smash_model_score": {
            "top": round(call.smash_score_top, 4),
            "bottom": round(call.smash_score_bottom, 4),
        },
        # Said on every call, in the payload, so no consumer can lose it.
        "score_is_calibrated": False,
        "score_kind": "uncalibrated model score",
        "score_note": MODEL_SCORE_STATEMENT,
    }
