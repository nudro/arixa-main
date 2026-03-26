"""Split selected images into evenly sized tiles.

Outputs are written under ./images/samples/<target_folder>.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
SAMPLES_DIR = IMAGES_DIR / "samples"

# filename -> (tile_count, output_folder_name)
SPLIT_PLAN: dict[str, tuple[int, str]] = {
    "Bella.png": (2, "bella"),
    "Gwen.png": (3, "Gwen"),
    "Mena.png": (9, "mena"),
    "Shay.png": (3, "shay"),
}


def factor_grid(count: int) -> tuple[int, int]:
    """Find rows, cols such that rows*cols=count and grid is near-square."""
    best_rows = 1
    best_cols = count
    best_delta = count - 1
    for rows in range(1, int(math.sqrt(count)) + 1):
        if count % rows != 0:
            continue
        cols = count // rows
        delta = abs(cols - rows)
        if delta < best_delta:
            best_rows, best_cols, best_delta = rows, cols, delta
    return best_rows, best_cols


def split_image_evenly(src_path: Path, tile_count: int, out_dir: Path) -> None:
    """Split one image into a deterministic rows x cols grid of tiles."""
    rows, cols = factor_grid(tile_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        width, height = img.size
        tile_w = width // cols
        tile_h = height // rows

        part_idx = 1
        stem = src_path.stem.lower()
        for r in range(rows):
            for c in range(cols):
                left = c * tile_w
                top = r * tile_h
                right = (c + 1) * tile_w if c < cols - 1 else width
                bottom = (r + 1) * tile_h if r < rows - 1 else height

                tile = img.crop((left, top, right, bottom))
                out_name = f"{stem}_{part_idx:02d}.png"
                tile.save(out_dir / out_name)
                part_idx += 1


def main() -> None:
    """Execute splits for all configured input files."""
    for filename, (count, folder_name) in SPLIT_PLAN.items():
        src_path = IMAGES_DIR / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source image: {src_path}")
        split_image_evenly(src_path, count, SAMPLES_DIR / folder_name)


if __name__ == "__main__":
    main()
