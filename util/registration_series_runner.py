"""Run deterministic registration for sample image series from CLI."""

from __future__ import annotations

import argparse
import json

from app.schemas.registration import RegistrationSeriesRequestSchema
from app.services.registration_service import register_image_series_from_samples


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sample series processing."""
    parser = argparse.ArgumentParser(description="Run MediaPipe registration on sample image series")
    parser.add_argument("--name", required=True, help="Subject folder under images/samples or images/sample")
    parser.add_argument(
        "--images",
        default="",
        help="Comma-separated image filenames. If omitted, processes all images in folder.",
    )
    parser.add_argument("--start-ts", default="2026-01-01T00:00:00Z", help="Synthetic series start timestamp")
    parser.add_argument("--step-seconds", type=int, default=60, help="Seconds between synthetic timestamps")
    return parser.parse_args()


def main() -> None:
    """Execute series registration and print final JSON result."""
    args = parse_args()
    image_names = [name.strip() for name in args.images.split(",") if name.strip()] or None
    request = RegistrationSeriesRequestSchema(
        subject_name=args.name,
        image_names=image_names,
        start_timestamp=args.start_ts,
        step_seconds=args.step_seconds,
    )
    result = register_image_series_from_samples(request)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
