"""
Batch inference using fine-tuned CatVTON (LoRA weights)
========================================================
Usage:
    python infer.py \
        --test_data    /workspace/dataset/test_data \
        --lora_weights /workspace/output/lora_mena/final \
        --output_dir   /workspace/test_results
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from model.attn_processor import SkipAttnProcessor
from model.pipeline import CatVTONPipeline
from model.utils import init_adapter
from peft import PeftModel
from utils import resize_and_crop, resize_and_padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_data",    required=True)
    p.add_argument("--lora_weights", required=True)
    p.add_argument("--output_dir",   default="./test_results")
    p.add_argument("--base_ckpt",    default="booksforcharlie/stable-diffusion-inpainting")
    p.add_argument("--attn_ckpt",    default="zhengchong/CatVTON")
    p.add_argument("--height",       type=int, default=1024)
    p.add_argument("--width",        type=int, default=768)
    p.add_argument("--steps",        type=int, default=50)
    p.add_argument("--guidance",     type=float, default=2.5)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    test_data = Path(args.test_data)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── Load pipeline (base SD-inpainting + CatVTON attention weights) ────────
    print("Loading pipeline ...")
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_ckpt,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version="mix",
        weight_dtype=torch.bfloat16,
        device=device,
        skip_safety_check=True,
    )

    # ── Load fine-tuned LoRA weights ──────────────────────────────────────────
    print(f"Loading LoRA weights from: {args.lora_weights}")
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_weights)
    pipeline.unet = pipeline.unet.merge_and_unload()
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)
    pipeline.unet.eval()
    print("LoRA merged.")

    # ── Mask processor (same as app.py — blurs mask for smooth edges) ─────────
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    # ── Load pairs ────────────────────────────────────────────────────────────
    image_dir = test_data / "image"
    cloth_dir = test_data / "cloth"
    mask_dir  = test_data / "agnostic-mask"
    size = (args.width, args.height)

    pairs_file = test_data / "test_pairs.txt"
    pairs = []
    if pairs_file.exists():
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        print(f"Loaded {len(pairs)} pairs from test_pairs.txt")
    else:
        for person_path in sorted(image_dir.iterdir()):
            stem = person_path.stem
            if not stem.endswith("_0"):
                continue
            base = stem[:-2]
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = cloth_dir / f"{base}_1{ext}"
                if candidate.exists():
                    pairs.append((person_path.name, candidate.name))
                    break
        print(f"Auto-matched {len(pairs)} pairs")

    # ── Inference loop ────────────────────────────────────────────────────────
    for person_name, cloth_name in tqdm(pairs, desc="Inference"):
        person_img = Image.open(image_dir / person_name).convert("RGB")
        cloth_img  = Image.open(cloth_dir  / cloth_name).convert("RGB")

        # Resize inputs (same as app.py)
        person_img = resize_and_crop(person_img, size)
        cloth_img  = resize_and_padding(cloth_img, size)

        # Load and prepare mask
        mask_fname = Path(person_name).stem + ".png"
        mask_path  = mask_dir / mask_fname
        if not mask_path.exists():
            # try same extension as person image
            mask_path = mask_dir / person_name
        mask_img = Image.open(mask_path)

        # Binarize: any non-black pixel → white (handles RGBA masks)
        mask_arr = np.array(mask_img.convert("L"))
        mask_arr[mask_arr > 0] = 255
        mask_img = Image.fromarray(mask_arr)
        mask_img = resize_and_crop(mask_img, size)

        # Blur mask edges (key step from app.py for smooth results)
        mask_img = mask_processor.blur(mask_img, blur_factor=9)

        # Run inference
        result = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask_img,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=generator,
        )[0]

        stem = Path(person_name).stem
        result.save(out_dir / f"{stem}_result.jpg", quality=95)

    print(f"\nDone! {len(pairs)} results saved to: {out_dir}")


if __name__ == "__main__":
    main()
