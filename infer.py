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
from PIL import Image, ImageFilter
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
    p.add_argument("--batch_size",   type=int, default=1,
                   help="Number of pairs to process per batch.")
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    p.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    p.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    return p.parse_args()

def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result
    
def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images
    
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
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
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
    for batch_start in tqdm(range(0, len(pairs), args.batch_size), desc="Inference"):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]

        person_imgs, cloth_imgs, mask_imgs, person_names = [], [], [], []
        for person_name, cloth_name in batch_pairs:
            person_img = Image.open(image_dir / person_name).convert("RGB")
            cloth_img  = Image.open(cloth_dir  / cloth_name).convert("RGB")

            # Resize inputs (same as app.py)
            person_img = resize_and_crop(person_img, size)
            cloth_img  = resize_and_padding(cloth_img, size)

            # Load and prepare mask
            mask_fname = Path(person_name).stem + ".png"
            mask_path  = mask_dir / mask_fname
            if not mask_path.exists():
                mask_path = mask_dir / person_name
            mask_img = Image.open(mask_path)

            # Binarize: any non-black pixel → white (handles RGBA masks)
            mask_arr = np.array(mask_img.convert("L"))
            mask_arr[mask_arr > 0] = 255
            mask_img = Image.fromarray(mask_arr)
            mask_img = resize_and_crop(mask_img, size)

            # Blur mask edges (key step from app.py for smooth results)
            mask_img = mask_processor.blur(mask_img, blur_factor=9)

            person_imgs.append(person_img)
            cloth_imgs.append(cloth_img)
            mask_imgs.append(mask_img)
            person_names.append(person_name)

        # Run inference one image at a time (pipeline expects single PIL images)
        results = []
        for person_img, cloth_img, mask_img in zip(person_imgs, cloth_imgs, mask_imgs):
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
            # pipeline may return a list or a single image
            results.append(result[0] if isinstance(result, list) else result)

        for i, result in enumerate(results):
            person_name = person_names[i]
            if args.repaint:
                mask_fname = Path(person_name).stem + ".png"
                mask_path  = mask_dir / mask_fname
                if not mask_path.exists():
                    mask_path = mask_dir / person_name
                original_person = Image.open(image_dir / person_name).convert("RGB").resize(result.size, Image.LANCZOS)
                original_mask   = Image.open(mask_path).resize(result.size, Image.NEAREST)
                result = repaint(original_person, original_mask, result)

            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new("RGB", (w * 3, h))
                concated_result.paste(person_imgs[i], (0, 0))
                concated_result.paste(cloth_imgs[i], (w, 0))
                concated_result.paste(result, (w * 2, 0))
                result = concated_result

            stem = Path(person_name).stem
            result.save(out_dir / f"{stem}_result.jpg", quality=95)

    print(f"\nDone! {len(pairs)} results saved to: {out_dir}")


if __name__ == "__main__":
    main()
