"""ROI polygon construction from MediaPipe face landmarks."""

from __future__ import annotations

from app.vision.mediapipe_landmarks import LandmarkPoint

Polygon = list[tuple[float, float]]


def build_roi_polygons(landmarks: list[LandmarkPoint]) -> dict[str, Polygon]:
    """Return canonical ROI polygons for facial regions used in Layer 1."""
    return {
        "cheeks": _points(landmarks, [50, 101, 234, 454, 330, 280]),
        "forehead": _points(landmarks, [10, 67, 103, 332, 297, 338]),
        "under_eye": _points(landmarks, [130, 133, 159, 386, 263, 362]),
        "upper_lip": _points(landmarks, [61, 185, 0, 267, 291, 39]),
        "temples": _points(landmarks, [127, 139, 356, 368]),
    }


def _points(landmarks: list[LandmarkPoint], idxs: list[int]) -> Polygon:
    """Return normalized (x, y) tuples for known indices when available."""
    pts: Polygon = []
    for idx in idxs:
        if idx < len(landmarks):
            p = landmarks[idx]
            pts.append((p.x, p.y))
    return pts
