"""
build_pairs.py — Build a clean train_pairs.txt for CatVTON LoRA fine-tuning
=============================================================================

Scans your dataset folder, validates every triplet (person image, cloth image,
agnostic mask), and writes only the pairs that are fully ready to use.

Checks performed per pair:
  1. Person image file exists
  2. Cloth image file exists
  3. Agnostic mask file exists  (expected: <person_stem>.png in agnostic-mask/)
  4. All three files can be opened by PIL (not corrupt)
  5. Mask is not all-black (mask generation failed) or all-white (no region found)

Usage:
    python build_pairs.py --data_root ./test_inputs
    python build_pairs.py --data_root ./test_inputs --output train_pairs_clean.txt
"""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Folder containing image/, cloth/, agnostic-mask/ and optionally train_pairs.txt")
    p.add_argument("--output", default="train_pairs.txt",
                   help="Output filename written inside data_root (default: train_pairs.txt)")
    p.add_argument("--min_mask_coverage", type=float, default=0.01,
                   help="Minimum fraction of mask pixels that must be white (default 0.01 = 1%%)")
    p.add_argument("--max_mask_coverage", type=float, default=0.95,
                   help="Maximum fraction of mask pixels that can be white (default 0.95 = 95%%)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print results without writing the output file")
    return p.parse_args()


def find_image_file(directory: Path, stem: str):
    """Find an image by stem, trying all supported extensions."""
    for ext in SUPPORTED_EXTENSIONS:
        for variant in [stem + ext, stem + ext.upper()]:
            candidate = directory / variant
            if candidate.exists():
                return candidate
    return None


def can_open(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


def check_mask(mask_path: Path, min_coverage: float, max_coverage: float):
    """
    Returns (ok, reason).
    ok=True means mask is usable.
    """
    try:
        mask = Image.open(mask_path).convert("L")
        arr = np.array(mask, dtype=np.float32) / 255.0
        coverage = arr.mean()
        if coverage < min_coverage:
            return False, f"mask all-black (coverage={coverage:.4f} < {min_coverage})"
        if coverage > max_coverage:
            return False, f"mask all-white (coverage={coverage:.4f} > {max_coverage})"
        return True, ""
    except Exception as e:
        return False, f"mask open error: {e}"


def collect_all_stems(image_dir: Path):
    """Return all person image stems found in image/."""
    stems = []
    for f in image_dir.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            stems.append(f.stem)
    return sorted(stems)


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    image_dir = data_root / "image"
    cloth_dir = data_root / "cloth"
    mask_dir  = data_root / "agnostic-mask"

    for d in [image_dir, cloth_dir, mask_dir]:
        if not d.exists():
            print(f"[ERROR] Required folder missing: {d}")
            return

    print(f"Scanning: {data_root}")
    print(f"  image/         — {sum(1 for f in image_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)} files")
    print(f"  cloth/         — {sum(1 for f in cloth_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)} files")
    print(f"  agnostic-mask/ — {sum(1 for f in mask_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)} files")
    print()

    stems = collect_all_stems(image_dir)
    print(f"Found {len(stems)} person images. Validating each triplet...\n")

    good_pairs = []
    skipped = []

    for stem in stems:
        # Find person image
        person_path = find_image_file(image_dir, stem)
        if person_path is None:
            skipped.append((stem, "person image file not found"))
            continue

        # Infer cloth stem: replace trailing _0 with _1
        if stem.endswith("_0"):
            cloth_stem = stem[:-2] + "_1"
        else:
            cloth_stem = stem  # fallback: same name

        cloth_path = find_image_file(cloth_dir, cloth_stem)
        if cloth_path is None:
            # Try same stem as person (some datasets use identical names)
            cloth_path = find_image_file(cloth_dir, stem)
        if cloth_path is None:
            skipped.append((stem, f"cloth image not found (tried stem '{cloth_stem}')"))
            continue

        # Mask is always <person_stem>.png
        mask_path = mask_dir / (stem + ".png")
        if not mask_path.exists():
            skipped.append((stem, "agnostic mask (.png) not found"))
            continue

        # Validate person image
        if not can_open(person_path):
            skipped.append((stem, f"corrupt person image: {person_path.name}"))
            continue

        # Validate cloth image
        if not can_open(cloth_path):
            skipped.append((stem, f"corrupt cloth image: {cloth_path.name}"))
            continue

        # Validate mask coverage
        mask_ok, mask_reason = check_mask(mask_path, args.min_mask_coverage, args.max_mask_coverage)
        if not mask_ok:
            skipped.append((stem, mask_reason))
            continue

        good_pairs.append((person_path.name, cloth_path.name))

    # Report
    print(f"Results:")
    print(f"  Valid pairs  : {len(good_pairs)}")
    print(f"  Skipped      : {len(skipped)}")

    if skipped:
        print(f"\nSkipped pairs and reasons:")
        for stem, reason in skipped:
            print(f"  {stem}: {reason}")

    if not good_pairs:
        print("\n[ERROR] No valid pairs found. Nothing written.")
        return

    output_path = data_root / args.output
    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for person_name, cloth_name in good_pairs:
                f.write(f"{person_name} {cloth_name}\n")
        print(f"\nWrote {len(good_pairs)} pairs -> {output_path}")
    else:
        print(f"\n[DRY RUN] Would write {len(good_pairs)} pairs -> {output_path}")
        print("First 10 pairs:")
        for pair in good_pairs[:10]:
            print(f"  {pair[0]}  {pair[1]}")


if __name__ == "__main__":
    main()
