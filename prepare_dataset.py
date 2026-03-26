"""
prepare_dataset.py — MENA Dataset Preparation for CatVTON LoRA Fine-Tuning
===========================================================================

Organizes your raw paired images into the VITON-HD training format,
validates pairs, and auto-generates cloth-agnostic masks.

Expected input structure (put your images here first):
    raw_data/
    ├── persons/        ← person photos WEARING the garment  (jpg/png)
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── garments/       ← flat-lay of the SAME garment (matching filename)
        ├── 001.jpg
        ├── 002.jpg
        └── ...

IMPORTANT: Filenames must match across both folders.
  persons/001.jpg  ↔  garments/001.jpg  →  one training pair
  persons/002.jpg  ↔  garments/002.jpg  →  another training pair

Output structure (ready for train_lora.py):
    output_dir/
    └── train/
        ├── image/              ← person images
        ├── cloth/              ← garment flat-lays
        ├── agnostic-mask/      ← auto-generated masks
        └── train_pairs.txt     ← pair list

Usage:
    # On a machine with GPU (cloud recommended):
    python prepare_dataset.py \\
        --raw_data_dir   ./raw_data \\
        --output_dir     ./data/mena \\
        --cloth_type     overall \\
        --repo_path      zhengchong/CatVTON

    # If models already downloaded locally:
    python prepare_dataset.py \\
        --raw_data_dir   ./raw_data \\
        --output_dir     ./data/mena \\
        --cloth_type     overall \\
        --repo_path      ./Models/CatVTON
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_dir", required=True,
                   help="Folder with persons/ and garments/ subfolders")
    p.add_argument("--output_dir",   required=True,
                   help="Where to write the structured dataset")
    p.add_argument("--cloth_type",   default="overall",
                   choices=["upper", "lower", "overall", "inner", "outer"],
                   help="Garment type — use 'overall' for abayas/full-length")
    p.add_argument("--repo_path",    default="zhengchong/CatVTON",
                   help="Local path or HuggingFace repo ID for CatVTON models")
    p.add_argument("--split",        default="train",
                   help="Split name (train or test)")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--skip_masks",   action="store_true",
                   help="Skip mask generation (useful if GPU not available locally)")
    return p.parse_args()


def find_image(folder: Path, stem: str):
    """Find an image file by stem regardless of extension."""
    for ext in SUPPORTED_EXTENSIONS:
        candidate = folder / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def collect_pairs(raw_dir: Path):
    """
    Scan persons/ and garments/ folders, match by filename stem.
    Returns list of (person_path, garment_path) tuples.
    """
    persons_dir  = raw_dir / "persons"
    garments_dir = raw_dir / "garments"

    assert persons_dir.exists(),  f"Missing: {persons_dir}"
    assert garments_dir.exists(), f"Missing: {garments_dir}"

    person_stems = {
        p.stem for p in persons_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    garment_stems = {
        g.stem for g in garments_dir.iterdir()
        if g.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    matched   = sorted(person_stems & garment_stems)
    only_person  = sorted(person_stems - garment_stems)
    only_garment = sorted(garment_stems - person_stems)

    if only_person:
        print(f"\n[WARN] {len(only_person)} person images have no matching garment "
              f"(will be skipped): {only_person[:5]}{'...' if len(only_person) > 5 else ''}")
    if only_garment:
        print(f"[WARN] {len(only_garment)} garment images have no matching person "
              f"(will be skipped): {only_garment[:5]}{'...' if len(only_garment) > 5 else ''}")

    pairs = []
    for stem in matched:
        person_path  = find_image(persons_dir,  stem)
        garment_path = find_image(garments_dir, stem)
        if person_path and garment_path:
            pairs.append((person_path, garment_path))

    return pairs


def copy_images(pairs, image_dir: Path, cloth_dir: Path):
    """
    Copy person/garment images to output directories.
    Returns list of (person_filename, cloth_filename) used in train_pairs.txt.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    cloth_dir.mkdir(parents=True, exist_ok=True)

    pair_names = []
    for person_path, garment_path in tqdm(pairs, desc="Copying images"):
        stem = person_path.stem

        # Always save as .jpg for consistency
        person_dst  = image_dir / f"{stem}.jpg"
        garment_dst = cloth_dir  / f"{stem}.jpg"

        if person_path.suffix.lower() == ".jpg":
            shutil.copy2(person_path, person_dst)
        else:
            Image.open(person_path).convert("RGB").save(person_dst, "JPEG", quality=95)

        if garment_path.suffix.lower() == ".jpg":
            shutil.copy2(garment_path, garment_dst)
        else:
            Image.open(garment_path).convert("RGB").save(garment_dst, "JPEG", quality=95)

        pair_names.append((f"{stem}.jpg", f"{stem}.jpg"))

    return pair_names


def write_pairs_txt(pair_names, output_path: Path):
    with open(output_path, "w") as f:
        for person_name, cloth_name in pair_names:
            f.write(f"{person_name} {cloth_name}\n")
    print(f"Wrote {len(pair_names)} pairs → {output_path}")


def generate_masks(pair_names, image_dir: Path, mask_dir: Path,
                   repo_path: str, cloth_type: str, device: str):
    """Run AutoMasker on each person image to generate cloth-agnostic masks."""
    from model.cloth_masker import AutoMasker

    mask_dir.mkdir(parents=True, exist_ok=True)

    # Download or use local model
    if os.path.exists(repo_path):
        densepose_ckpt = os.path.join(repo_path, "DensePose")
        schp_ckpt      = os.path.join(repo_path, "SCHP")
    else:
        print(f"Downloading CatVTON models from {repo_path} ...")
        local_repo = snapshot_download(repo_id=repo_path)
        densepose_ckpt = os.path.join(local_repo, "DensePose")
        schp_ckpt      = os.path.join(local_repo, "SCHP")

    automasker = AutoMasker(
        densepose_ckpt=densepose_ckpt,
        schp_ckpt=schp_ckpt,
        device=device,
    )

    skipped = 0
    for person_name, _ in tqdm(pair_names, desc="Generating masks"):
        stem     = Path(person_name).stem
        mask_dst = mask_dir / f"{stem}.png"

        if mask_dst.exists():
            continue  # already done

        person_img_path = str(image_dir / person_name)
        try:
            mask = automasker(person_img_path, cloth_type)["mask"]
            mask.save(mask_dst)
        except Exception as e:
            print(f"  [WARN] Mask generation failed for {person_name}: {e}")
            skipped += 1

    if skipped:
        print(f"\n[WARN] {skipped} masks failed — those pairs will be skipped during training.")
    print(f"Masks saved → {mask_dir}")


def verify_dataset(split_dir: Path, pair_names):
    """Quick sanity check: confirm all expected files exist."""
    image_dir = split_dir / "image"
    cloth_dir = split_dir / "cloth"
    mask_dir  = split_dir / "agnostic-mask"

    missing = []
    for person_name, cloth_name in pair_names:
        stem = Path(person_name).stem
        for path in [
            image_dir / person_name,
            cloth_dir / cloth_name,
            mask_dir  / f"{stem}.png",
        ]:
            if not path.exists():
                missing.append(str(path))

    if missing:
        print(f"\n[WARN] {len(missing)} expected files not found:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("\n✓ Dataset verified — all images and masks present.")


def main():
    args = parse_args()
    raw_dir    = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    split_dir  = output_dir / args.split

    print(f"\n{'='*55}")
    print(f"  CatVTON Dataset Preparation")
    print(f"  Input  : {raw_dir}")
    print(f"  Output : {split_dir}")
    print(f"  Type   : {args.cloth_type}")
    print(f"{'='*55}\n")

    # Step 1: match pairs
    print("Step 1/4 — Scanning for matched pairs ...")
    pairs = collect_pairs(raw_dir)
    print(f"  Found {len(pairs)} matched pairs.")

    if len(pairs) == 0:
        print("\n[ERROR] No matched pairs found. Check that persons/ and garments/ "
              "exist and share filenames.")
        return

    if len(pairs) < 100:
        print(f"\n[WARN] Only {len(pairs)} pairs found. Recommended minimum is 300–500 "
              f"for meaningful fine-tuning.")

    # Step 2: copy images
    print("\nStep 2/4 — Copying images ...")
    pair_names = copy_images(
        pairs,
        image_dir=split_dir / "image",
        cloth_dir=split_dir / "cloth",
    )

    # Step 3: write pairs txt
    print("\nStep 3/4 — Writing train_pairs.txt ...")
    write_pairs_txt(pair_names, split_dir / "train_pairs.txt")

    # Step 4: generate masks
    if args.skip_masks:
        print("\nStep 4/4 — Skipping mask generation (--skip_masks set).")
        print("  Run this later on a GPU machine:")
        print(f"  python prepare_dataset.py --raw_data_dir {raw_dir} "
              f"--output_dir {output_dir} --cloth_type {args.cloth_type} "
              f"--repo_path {args.repo_path}")
    else:
        print("\nStep 4/4 — Generating cloth-agnostic masks (requires GPU) ...")
        generate_masks(
            pair_names,
            image_dir=split_dir / "image",
            mask_dir=split_dir / "agnostic-mask",
            repo_path=args.repo_path,
            cloth_type=args.cloth_type,
            device=args.device,
        )

    # Verify
    if not args.skip_masks:
        print("\nVerifying dataset ...")
        verify_dataset(split_dir, pair_names)

    print(f"\n{'='*55}")
    print(f"  Done! Dataset ready at: {split_dir}")
    print(f"  Total pairs: {len(pair_names)}")
    print(f"\n  Next step — run training:")
    print(f"    python train_lora.py \\")
    print(f"        --data_root {output_dir} \\")
    print(f"        --cloth_type {args.cloth_type} \\")
    print(f"        --output_dir ./output/lora_mena")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
