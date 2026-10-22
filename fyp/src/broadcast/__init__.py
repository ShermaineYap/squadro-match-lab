"""Broadcast stroke analysis — an **experimental** second video mode.

The movement stream measures a single athlete from a side-on camera and names
the movement by temporal rules over MediaPipe landmarks. This package does a
different job on a different kind of footage: **one short rally** from a fixed
rear-court **broadcast** recording, both players in frame, the kind BWF
publishes. On that footage the pose-only route cannot separate a smash from a
clear or a drop — the project established that twice — so this module adds the
one input that can: the shuttlecock's trajectory.

Pipeline, all pretrained, nothing trained here:

    video ─▶ YOLO11 pose (both players, COCO-17)
          ─▶ TrackNetV3 (shuttle trajectory)
          ─▶ hits = reversals of the shuttle's vertical direction
          ─▶ BST_0 (pose + shuttle) per between-hits window ─▶ stroke type
          ─▶ lowest-foot vertical pulse of the hitter around the hit ─▶ jump evidence
          ─▶ fusion: smash + pulse ⇒ "likely jump smash"

Boundaries, stated once here and repeated in every payload:

* **Experimental and unvalidated.** Not used in the movement result, the policy
  grades, the official results, the audit record or the report of record. Off
  the API unless ``ENABLE_BROADCAST_STROKE_ANALYSIS=on`` and its optional
  dependencies are installed, and never load-bearing when they are. Not
  production ready.
* **No accuracy figure, on this footage or any other.** BST_0's weights come
  from ShuttleSet (broadcast singles, professional play). This project has no
  labelled broadcast footage, so what is reported is the classifier's softmax:
  an **uncalibrated model score**, not a confidence and not a probability that
  the call is right. ``docs/BROADCAST_EVALUATION_PROTOCOL.md`` is the protocol
  that would produce a real figure; it has not been run.
* **Jump evidence is an image-space indicator.** A rise-and-return of the
  hitter's lowest foot in the picture, in torso lengths. Not a jump height, not
  a measurement of flight, and an unmeasurable pulse is never reported as the
  absence of a jump.
* **The pose estimator was substituted.** BST was developed using MMPose-derived
  inputs. This experimental adapter substitutes YOLO11 COCO-17 landmarks.
  Matching landmark topology does not establish matching error characteristics
  or equivalent model performance.
* **Third-party models, cited not claimed.** BST — Chang, "BST: Badminton
  Stroke-type Transformer for Skeleton-based Action Recognition in Racket
  Sports", CVPRW 2026 (MIT). TrackNetV3 — qaz812345 (MIT). YOLO pose —
  Ultralytics (AGPL-3.0; academic, local use). None of these is redistributed
  in this repository; ``scripts/setup_broadcast_module.py`` fetches them at
  pinned commits and verifies every weight file by SHA-256.
"""

from src.broadcast.third_party import dependency_status  # noqa: F401
