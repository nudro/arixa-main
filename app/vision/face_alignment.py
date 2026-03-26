"""Deterministic face-angle classification and alignment metadata helpers."""

from __future__ import annotations

from typing import TypedDict

from app.vision.mediapipe_landmarks import LandmarkPoint


class AlignmentTransform(TypedDict):
    """Alignment transform metadata for downstream warping steps."""

    center_x: float
    center_y: float
    roll_degrees: float
    scale: float


def classify_angle(landmarks: list[LandmarkPoint]) -> str:
    """Classify pose into coarse angle buckets from landmark geometry."""
    if len(landmarks) < 455:
        return "unknown"

    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    symmetry = ((nose.x - left_cheek.x) - (right_cheek.x - nose.x))

    if abs(symmetry) < 0.03:
        return "front"
    if symmetry <= -0.12:
        return "left_profile"
    if symmetry >= 0.12:
        return "right_profile"
    if symmetry < 0:
        return "left_45"
    return "right_45"


def estimate_alignment_transform(landmarks: list[LandmarkPoint]) -> AlignmentTransform:
    """Estimate transform metadata from eye centers and inter-eye distance."""
    if len(landmarks) < 264:
        return AlignmentTransform(center_x=0.5, center_y=0.5, roll_degrees=0.0, scale=1.0)

    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    roll_degrees = -57.2958 * _atan2(dy, dx)
    scale = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    center_x = (left_eye.x + right_eye.x) / 2.0
    center_y = (left_eye.y + right_eye.y) / 2.0
    return AlignmentTransform(
        center_x=center_x,
        center_y=center_y,
        roll_degrees=roll_degrees,
        scale=scale,
    )


def _atan2(y: float, x: float) -> float:
    """Small wrapper for atan2 to keep pure unit-testable math path."""
    import math

    return math.atan2(y, x)
