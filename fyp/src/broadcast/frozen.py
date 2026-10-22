"""Frozen parameters, frozen provenance, and the sentences that must not drift."""

from __future__ import annotations

from typing import Any

FROZEN_ON = "2026-08-25"

DEVELOPMENT_FOOTAGE = (
    "One 10.5 s rally (An Se Young v Akane Yamaguchi, BWF highlights upload, "
    "1280x720, 25 fps, frames 0-262, fixed rear-court view). Development and "
    "feasibility evidence only; never to be reused as evaluation data."
)


# --------------------------------------------------------------------------
# The five frozen thresholds
# --------------------------------------------------------------------------

STROKE_MODEL_SCORE_THRESHOLD = 0.5

FOOT_PULSE_TORSOS = 0.10

JUMP_WINDOW_SECONDS = 0.4

MIN_HIT_SEPARATION_SECONDS = 0.4

SHUTTLE_REVERSAL_PX = 1.5

SHUTTLE_SMOOTHING_FRAMES = 5

FROZEN_THRESHOLDS: dict[str, dict[str, Any]] = {
    "stroke_model_score_threshold": {
        "value": STROKE_MODEL_SCORE_THRESHOLD,
        "unit": "uncalibrated model score (softmax output)",
        "role": "below this the classifier's top class is not taken as a call",
    },
    "foot_pulse_torsos": {
        "value": FOOT_PULSE_TORSOS,
        "unit": "torso lengths, image-space vertical",
        "role": "size of the lowest-foot rise-and-return that reads as jump evidence",
    },
    "jump_window_seconds": {
        "value": JUMP_WINDOW_SECONDS,
        "unit": "seconds either side of the hit",
        "role": "window the foot pulse is searched in",
    },
    "min_hit_separation_seconds": {
        "value": MIN_HIT_SEPARATION_SECONDS,
        "unit": "seconds",
        "role": "minimum gap between two accepted hits",
    },
    "shuttle_reversal_px": {
        "value": SHUTTLE_REVERSAL_PX,
        "unit": "pixels per frame",
        "role": "smallest vertical-velocity change accepted as a reversal",
    },
    "shuttle_smoothing_frames": {
        "value": SHUTTLE_SMOOTHING_FRAMES,
        "unit": "frames",
        "role": "centred moving average applied before differentiating",
    },
}

THRESHOLDS_STATEMENT = (
    "These are exploratory development choices, frozen on "
    f"{FROZEN_ON} before any independent evaluation footage was analysed. They "
    "were informed by a single development rally and by a threshold carried over "
    "unchanged from the movement stream; none of them was fitted to labelled "
    "data, and no operating point was selected from a curve. They must not be "
    "tuned on the independent evaluation set: doing so would make that set a "
    "second development set and would invalidate every number measured on it. "
    "If a threshold has to change, the change and the reason are recorded here "
    "and the evaluation is re-run from the beginning on fresh footage."
)


# --------------------------------------------------------------------------
# The statements
# --------------------------------------------------------------------------

MMPOSE_SUBSTITUTION_STATEMENT = (
    "BST was developed using MMPose-derived inputs. This experimental adapter "
    "substitutes YOLO11 COCO-17 landmarks. Matching landmark topology does not "
    "establish matching error characteristics or equivalent model performance."
)

MODEL_SCORE_STATEMENT = (
    "The number beside each stroke is an uncalibrated model score - the "
    "classifier's raw softmax output on footage from a camera and a pose "
    "estimator it was not trained on. It is not a confidence, not a calibrated "
    "probability, and not an estimate of how often the call is right."
)

FOOT_PULSE_STATEMENT = (
    "Jump evidence is a lowest-foot vertical pulse: the hitter's lower ankle "
    "rises in the image and returns, inside the window. This is an image-space "
    "indicator measured in torso lengths, not a physical jump height and not a "
    "measurement of flight. A pulse below the threshold means the indicator did "
    "not fire; it is not proof the player stayed on the ground. An unmeasurable "
    "pulse means nothing was evaluated at all."
)

RALLY_CONTRACT = (
    "Fixed rear-court camera view for the whole analysed range.",
    "One rally only. This is not full-match analysis.",
    "No replay, no close-up, no camera cut inside the range.",
    "Both players and the shuttle visible throughout.",
)

RALLY_CONTRACT_STATEMENT = (
    "This module analyses one short rally from a fixed rear-court camera. It is "
    "not full-match analysis and does not segment a match into rallies. Nothing "
    "in the pipeline can detect a replay, a close-up or a camera cut: footage "
    "that breaks the contract is tracked as though it were court, and the "
    "output will be wrong without saying so."
)

MERGED_STROKE_CLASSES: tuple[str, ...] = (
    "未知球種",
    "Top_放小球", "Top_擋小球", "Top_殺球", "Top_挑球",
    "Top_長球", "Top_平球", "Top_切球", "Top_推球",
    "Top_撲球", "Top_勾球", "Top_發短球", "Top_發長球",
    "Bottom_放小球", "Bottom_擋小球", "Bottom_殺球", "Bottom_挑球",
    "Bottom_長球", "Bottom_平球", "Bottom_切球", "Bottom_推球",
    "Bottom_撲球", "Bottom_勾球", "Bottom_發短球", "Bottom_發長球",
)


# --------------------------------------------------------------------------
# Diagnostic footage — spent, like the development rally
# --------------------------------------------------------------------------

DIAGNOSTIC_FOOTAGE = (
    "One 16.03 s rally analysed twice, at 1280x720 and at 1920x1080, 30 fps, "
    "source frames 10950-11430, fixed rear-court view (men's singles, "
    "scoreboard reads SHI Y.Q. v AXELSEN, Odense). Run on 2026-08-26 to "
    "separate resolution effects from classifier behaviour. Diagnostic evidence "
    "only: the foot-pulse rejection below was taken from it, so it is spent in "
    "the same way the development rally is and must not appear in the "
    "evaluation set or in any accuracy claim."
)

DIAGNOSTIC_FINDINGS = (
    "Both-players tracking rose from 54% at 720p to 100% at 1080p, and "
    "determined strokes from 2 of 22 to 10 of 23. Hit detection was almost "
    "resolution-invariant: 21 hits paired within 0.25 s and the last 16 hit "
    "times were identical to the frame. The classifier was not: on those "
    "paired hits the top class agreed only 11 times out of 21, and both of the "
    "two calls the 720p run did report were contradicted by the 1080p run. "
    "Neither run's stroke types have been checked against a human label, so "
    "nothing here is an accuracy measurement in either direction."
)


# --------------------------------------------------------------------------
# The foot-pulse indicator: rejected before evaluation
# --------------------------------------------------------------------------

JUMP_EVIDENCE_REJECTED = True

JUMP_EVIDENCE_REJECTED_ON = "2026-08-26"

JUMP_EVIDENCE_REJECTION_STATEMENT = (
    "The lowest-foot pulse indicator was rejected on 2026-08-26, on diagnostic "
    "footage, before any evaluation clip was opened. It is excluded from the "
    "formal evaluation and from the displayed result. It is retained in the "
    "payload and in the documentation as a recorded unsuccessful experiment, "
    "and no precision, recall or accuracy is claimed or will be measured for "
    "it. Rejecting it costs the module nothing it was measuring: the stroke "
    "type has never been a function of the pulse, and a smash is reported as a "
    "smash whatever the indicator did."
)

JUMP_EVIDENCE_REJECTION_EVIDENCE = (
    "On the 1080p diagnostic rally the indicator fired five times, at 0.412, "
    "0.480, 0.929, 1.060 and 1.136 torso lengths, against non-firings of 0.000 "
    "to 0.092 torso lengths. Nothing was measured between 0.092 and 0.412, so "
    "the 0.10 threshold sits in an empty region rather than between two "
    "populations. A lowest ankle rising and returning more than a whole torso "
    "length inside the 0.8 s window is not a badminton jump; it is the ankle "
    "keypoint moving between legs or between players. The indicator fired on "
    "none of the four hits the classifier called a smash, and three of its five "
    "firings were on hits the validity gate had already declined. The fused "
    "state likely_jump_smash was produced zero times."
)

FUSION_STATES_REJECTED_STATEMENT = (
    "The six fusion states are retained experimental record, not a result. "
    "They combine the classifier's class with the rejected foot-pulse "
    "indicator, so any state that mentions a jump inherits that rejection. The "
    "module's result is the stroke type and the hit it belongs to."
)


# --------------------------------------------------------------------------
# Player boxes: a detection, not an identification
# --------------------------------------------------------------------------

PLAYER_BOX_STATEMENT = (
    "These boxes are the pose estimator's person detections for the frame, "
    "ordered [top, bottom] by which side of the image line at "
    "NET_LINE_FRACTION a detection's centre falls on and, within a side, by "
    "area. They are not player identification: nobody is recognised, no name "
    "is attached, and the ordering can swap or drop a player where both are "
    "on the same side of that line, where one is occluded, or where an umpire "
    "or line judge is the larger detection. They are not a court calibration "
    "either - there is no homography, so 'top' means higher in the picture "
    "and nothing more. A stroke's hitter side is derived from the "
    "classifier's own label and not from these boxes, so the two are "
    "independent and can disagree; where they do, neither is corrected to "
    "match the other, and a box drawn around a player is not evidence that "
    "the player struck the shuttle."
)


SHUTTLE_TRACK_STATEMENT = (
    "One point per frame: where TrackNetV3 placed the shuttle, in source-video "
    "pixels. It is a position, not an extent - the tracker outputs a centre "
    "and no size, so the marker drawn around it is a fixed drawing convention "
    "and its size measures nothing. A frame with no point is a frame the "
    "tracker lost the shuttle in, which is not the same as a frame the shuttle "
    "was absent from, and it is drawn as nothing rather than interpolated "
    "between the points either side. The tracker is pretrained and its "
    "accuracy has not been measured on this footage or any other."
)


# --------------------------------------------------------------------------
# The hitter side is not a second measurement
# --------------------------------------------------------------------------

HITTER_SIDE_DERIVED_STATEMENT = (
    "The hitter side is derived, not separately predicted. BST_0's classes name "
    "the side and the stroke jointly (Top_smash, Bottom_smash, ...), and the "
    "reported hitter is that label's prefix. Side accuracy is therefore a "
    "coarsening of stroke accuracy over the same softmax: it cannot be wrong "
    "where the stroke is right, and its errors are a subset of the stroke's. It "
    "must be reported as a derived figure and never as a second, independent "
    "model or capability."
)
