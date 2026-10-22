"""The two players, per frame, in the order BST expects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.broadcast.third_party import WEIGHTS_DIR, YOLO_POSE_WEIGHT

PERSON_CONFIDENCE = 0.35
NET_LINE_FRACTION = 0.52


@dataclass
class PlayerTracks:
    """Per-frame keypoints and boxes for [top, bottom]; zeros where absent."""

    keypoints: np.ndarray  # (t, 2, 17, 2) pixels
    boxes: np.ndarray  # (t, 2, 4) xyxy pixels
    width: int
    height: int
    fps: float

    @property
    def both_present(self) -> np.ndarray:
        return self.boxes[:, 0].any(axis=1) & self.boxes[:, 1].any(axis=1)


def extract_players(
    video: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    on_frame: Any = None,
) -> PlayerTracks:
    """Track both players frame by frame.

    ``on_frame(done, total)`` is called after each frame, if given. This loop is
    the only part of the pose stage the caller can see inside, so it is where a
    progress bar gets a real position rather than an invented one. It changes no
    detection, no ordering and no output.
    """
    import cv2
    from ultralytics import YOLO

    # Ultralytics downloads into the working directory when handed a bare name;
    # handing it a path under models/ keeps that download where the other
    # weights live. The file is fetched on first use if absent.
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_POSE_WEIGHT) if YOLO_POSE_WEIGHT.exists() else YOLO_POSE_WEIGHT.name)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("The clip could not be opened.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = total - 1 if end_frame is None else min(end_frame, total - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    keypoints: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    wanted = max(1, last - start_frame + 1)
    try:
        for _ in range(start_frame, last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            kp = np.zeros((2, 17, 2))
            bb = np.zeros((2, 4))
            result = model.predict(frame, verbose=False, conf=PERSON_CONFIDENCE)[0]
            if result.boxes is not None and len(result.boxes) >= 2 and result.keypoints is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                pts = result.keypoints.xy.cpu().numpy()
                centre_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
                area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                far = [i for i in range(len(xyxy)) if centre_y[i] < height * NET_LINE_FRACTION]
                near = [i for i in range(len(xyxy)) if centre_y[i] >= height * NET_LINE_FRACTION]
                if far and near:
                    top = max(far, key=lambda i: area[i])
                    bottom = max(near, key=lambda i: area[i])
                    kp[0], bb[0] = pts[top], xyxy[top]
                    kp[1], bb[1] = pts[bottom], xyxy[bottom]
            keypoints.append(kp)
            boxes.append(bb)
            if on_frame is not None:
                on_frame(len(keypoints), wanted)
    finally:
        capture.release()

    if not keypoints:
        raise RuntimeError("No frames could be read from the clip.")
    return PlayerTracks(np.stack(keypoints), np.stack(boxes), width, height, float(fps))


def yolo_weight_present() -> bool:
    return YOLO_POSE_WEIGHT.exists()


def as_summary(tracks: PlayerTracks) -> dict[str, Any]:
    return {
        "frames": int(len(tracks.keypoints)),
        "both_players_fraction": round(float(tracks.both_present.mean()), 3),
        "resolution": [tracks.width, tracks.height],
        "fps": round(tracks.fps, 3),
        "pose_estimator": "Ultralytics YOLO11 pose (COCO-17)",
    }
