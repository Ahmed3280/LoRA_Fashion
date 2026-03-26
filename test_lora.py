"""
Batch inference for LoRA fine-tuned CatVTON
============================================

Expected test_data structure (two layouts supported):

  Layout A — with pairs file:
    test_data/
        image/              <- person images
        cloth/              <- garment images
        agnostic-mask/      <- (optional) pre-generated masks
        test_pairs.txt      <- "person.jpg cloth.jpg" per line

  Layout B — no pairs file (auto-matches by filename stem):
    test_data/
        image/              <- person images  e.g. 001_0.jpg
        cloth/              <- garment images e.g. 001_1.jpg
        agnostic-mask/      <- (optional) pre-generated masks

Usage:
    python test_lora.py \
        --test_data     /workspace/dataset/test_data \
        --lora_weights  /workspace/output/lora_mena/final \
        --output_dir    /workspace/test_results \
        --cloth_type    overall
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_data",    required=True,
                   help="Path to test_data folder (contains image/ and cloth/)")
    p.add_argument("--lora_weights", required=True,
                   help="Path to LoRA checkpoint dir (local) or HF repo id")
    p.add_argument("--base_ckpt",    default="booksforcharlie/stable-diffusion-inpainting")
    p.add_argument("--attn_ckpt",    default="zhengchong/CatVTON")
    p.add_argument("--attn_ckpt_version", default="mix",
                   choices=["mix", "vitonhd", "dresscode"])
    p.add_argument("--output_dir",   default="./test_results")
    p.add_argument("--cloth_type",   default="overall",
                   choices=["upper", "lower", "overall", "inner", "outer"])
    p.add_argument("--height",       type=int, default=1024)
    p.add_argument("--width",        type=int, default=768)
    p.add_argument("--steps",        type=int, default=50)
    p.add_argument("--guidance",     type=float, default=2.5)
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--no_automasker", action="store_true",
                   help="Skip AutoMasker and use pre-existing masks in agnostic-mask/")
    return p.parse_args()


def load_pairs(test_data: Path):
    """Return list of (person_name, cloth_name) pairs."""
    pairs_file = test_data / "test_pairs.txt"
    if pairs_file.exists():
        pairs = []
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        print(f"Loaded {len(pairs)} pairs from test_pairs.txt")
        return pairs

    # Auto-match: person files end with _0, cloth files end with _1
    image_dir = test_data / "image"
    cloth_dir = test_data / "cloth"
    pairs = []
    for person_path in sorted(image_dir.iterdir()):
        stem = person_path.stem          # e.g. "001_0"
        if not stem.endswith("_0"):
            continue
        base = stem[:-2]                 # e.g. "001"
        # find matching cloth (any extension)
        cloth_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = cloth_dir / f"{base}_1{ext}"
            if candidate.exists():
                cloth_path = candidate
                break
        if cloth_path:
            pairs.append((person_path.name, cloth_path.name))

    if not pairs:
        # Fallback: just pair every image with the same-name cloth
        for person_path in sorted(image_dir.iterdir()):
            cloth_path = cloth_dir / person_path.name
            if cloth_path.exists():
                pairs.append((person_path.name, person_path.name))

    print(f"Auto-matched {len(pairs)} pairs (no test_pairs.txt found)")
    return pairs


def main():
    args = parse_args()
    test_data = Path(args.test_data)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = {"no": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── Load CatVTON pipeline ─────────────────────────────────────────────────
    print("Loading CatVTON pipeline ...")
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_ckpt,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version=args.attn_ckpt_version,
        weight_dtype=weight_dtype,
        device=device,
        skip_safety_check=True,
    )

    # ── Apply LoRA weights ────────────────────────────────────────────────────
    print(f"Loading LoRA weights from: {args.lora_weights}")
    from peft import PeftModel
    from model.attn_processor import SkipAttnProcessor
    from model.utils import init_adapter
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_weights)
    pipeline.unet = pipeline.unet.merge_and_unload()   # fuse LoRA into weights for faster inference
    # Re-apply SkipAttnProcessor — merge_and_unload() resets attention processors
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)
    pipeline.unet.eval()
    print("LoRA weights merged.")

    # ── Load AutoMasker (for pairs without pre-generated masks) ───────────────
    mask_dir = test_data / "agnostic-mask"
    need_automasker = (not args.no_automasker) and (not mask_dir.exists() or not any(mask_dir.iterdir()))
    if need_automasker:
        print("No pre-generated masks found — loading AutoMasker ...")
        automasker = AutoMasker(
            densepose_ckpt="./Models/DensePose",
            schp_ckpt="./Models/SCHP",
            device=device,
        )
    else:
        automasker = None
        print(f"Using pre-generated masks from: {mask_dir}")

    # ── Load pairs ────────────────────────────────────────────────────────────
    pairs = load_pairs(test_data)
    if not pairs:
        print("ERROR: No pairs found in test_data. Check your folder structure.")
        return

    image_dir = test_data / "image"
    cloth_dir = test_data / "cloth"
    size = (args.width, args.height)

    # ── Inference loop ────────────────────────────────────────────────────────
    for person_name, cloth_name in tqdm(pairs, desc="Generating"):
        person_path = image_dir / person_name
        cloth_path  = cloth_dir / cloth_name

        person_img = Image.open(person_path).convert("RGB")
        cloth_img  = Image.open(cloth_path).convert("RGB")

        # Get or generate mask
        if automasker is None:
            mask_fname = Path(person_name).stem + ".png"
            mask_path  = mask_dir / mask_fname
            if mask_path.exists():
                mask_img = Image.open(mask_path).convert("L")
            else:
                # Try same extension
                mask_path2 = mask_dir / person_name
                if mask_path2.exists():
                    mask_img = Image.open(mask_path2).convert("L")
                else:
                    print(f"  [WARN] No mask for {person_name}, generating on-the-fly ...")
                    mask_result = automasker(
                        resize_and_crop(person_img, size), args.cloth_type
                    )
                    mask_img = mask_result["mask"]
        else:
            mask_result = automasker(
                resize_and_crop(person_img, size), args.cloth_type
            )
            mask_img = mask_result["mask"]

        # Run pipeline
        result = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask_img,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        output_img = result[0]   # PIL Image

        stem = Path(person_name).stem

        # Save try-on result only
        output_img.save(out_dir / f"{stem}_result.jpg", quality=95)

        # Save comparison (person | cloth | result) for reference
        person_resized = resize_and_crop(person_img, size)
        cloth_resized  = resize_and_padding(cloth_img, size)
        canvas = Image.new("RGB", (args.width * 3, args.height))
        canvas.paste(person_resized, (0, 0))
        canvas.paste(cloth_resized,  (args.width, 0))
        canvas.paste(output_img,     (args.width * 2, 0))
        canvas.save(out_dir / f"{stem}_compare.jpg", quality=95)

    print(f"\nDone! Results saved to: {out_dir}")
    print(f"  *_result.jpg  — try-on output only")
    print(f"  *_compare.jpg — [person | garment | result] side-by-side")


if __name__ == "__main__":
    main()
