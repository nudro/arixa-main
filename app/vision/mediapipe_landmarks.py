"""MediaPipe-based static-image face detection and landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class LandmarkPoint:
    """One normalized landmark point."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceLandmarkResult:
    """Structured face detection and landmark extraction output."""

    face_detected: bool
    landmarks: list[LandmarkPoint]
    landmarks_count: int
    bbox_px: tuple[int, int, int, int]
    warnings: list[str]


def extract_face_landmarks(image_path: str) -> tuple[np.ndarray, FaceLandmarkResult]:
    """Load local image path and return image array plus primary-face landmarks."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return _empty_image_result("image_not_found_or_unreadable")

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        h, w = image_bgr.shape[:2]
        return image_bgr, FaceLandmarkResult(
            face_detected=False,
            landmarks=[],
            landmarks_count=0,
            bbox_px=(0, 0, w, h),
            warnings=["mediapipe_face_mesh_unavailable"],
        )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as mesh:
        result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        h, w = image_bgr.shape[:2]
        return image_bgr, FaceLandmarkResult(
            face_detected=False,
            landmarks=[],
            landmarks_count=0,
            bbox_px=(0, 0, w, h),
            warnings=["face_not_detected"],
        )

    face_landmarks = result.multi_face_landmarks[0].landmark
    points = [LandmarkPoint(x=p.x, y=p.y, z=p.z) for p in face_landmarks]
    bbox = _landmark_bbox(points, image_bgr.shape[1], image_bgr.shape[0])
    return image_bgr, FaceLandmarkResult(
        face_detected=True,
        landmarks=points,
        landmarks_count=len(points),
        bbox_px=bbox,
        warnings=[],
    )


def _landmark_bbox(points: list[LandmarkPoint], width: int, height: int) -> tuple[int, int, int, int]:
    """Compute pixel bounding box from normalized landmark points."""
    xs = [min(max(p.x, 0.0), 1.0) for p in points]
    ys = [min(max(p.y, 0.0), 1.0) for p in points]
    min_x, max_x = int(min(xs) * width), int(max(xs) * width)
    min_y, max_y = int(min(ys) * height), int(max(ys) * height)
    return (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))


def _empty_image_result(reason: str) -> tuple[np.ndarray, FaceLandmarkResult]:
    """Return empty default payload for unreadable image paths."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    return image, FaceLandmarkResult(
        face_detected=False,
        landmarks=[],
        landmarks_count=0,
        bbox_px=(0, 0, 1, 1),
        warnings=[reason],
    )
