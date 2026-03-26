"""Unit tests for ROI polygon generation from landmarks."""

from app.vision.mediapipe_landmarks import LandmarkPoint
from app.vision.roi_regions import build_roi_polygons


def test_build_roi_polygons_contains_expected_regions() -> None:
    """ROI builder returns all configured region keys."""
    landmarks = [LandmarkPoint(x=0.5, y=0.5, z=0.0) for _ in range(500)]
    rois = build_roi_polygons(landmarks)
    assert set(rois.keys()) == {"cheeks", "forehead", "under_eye", "upper_lip", "temples"}
    assert all(len(poly) > 0 for poly in rois.values())
