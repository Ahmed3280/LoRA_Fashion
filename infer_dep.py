"""
Single-image deployment inference for fine-tuned CatVTON (LoRA weights).

Usage:
    python infer_dep.py \
        --person     path/to/person.jpg \
        --cloth      path/to/cloth.jpg \
        --mask       path/to/mask.png \
        --lora_path  path/to/lora_weights \
        --output     result.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from model.attn_processor import SkipAttnProcessor
from model.pipeline import CatVTONPipeline
from model.utils import init_adapter
from peft import PeftModel
from utils import resize_and_crop, resize_and_padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--person",    required=True, help="Path to person image")
    p.add_argument("--cloth",     required=True, help="Path to cloth image")
    p.add_argument("--mask",      required=True, help="Path to agnostic mask")
    p.add_argument("--lora_path", required=True, help="Folder with fine-tuned LoRA weights")
    p.add_argument("--output",    default="result.jpg", help="Output image path")
    p.add_argument("--base_ckpt", default="booksforcharlie/stable-diffusion-inpainting")
    p.add_argument("--attn_ckpt", default="zhengchong/CatVTON")
    p.add_argument("--height",    type=int, default=1024)
    p.add_argument("--width",     type=int, default=768)
    p.add_argument("--steps",     type=int, default=50)
    p.add_argument("--guidance",  type=float, default=2.5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load pipeline (base SD-inpainting + CatVTON attention weights)
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

    # Merge fine-tuned LoRA weights (identical to infer.py)
    print(f"Loading LoRA weights from: {args.lora_path}")
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_path)
    pipeline.unet = pipeline.unet.merge_and_unload()
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)
    pipeline.unet.eval()
    print("LoRA merged.")

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    size = (args.width, args.height)

    # Load & preprocess inputs
    person_img = Image.open(args.person).convert("RGB")
    cloth_img  = Image.open(args.cloth).convert("RGB")
    person_img = resize_and_crop(person_img, size)
    cloth_img  = resize_and_padding(cloth_img, size)

    mask_img = Image.open(args.mask)
    mask_arr = np.array(mask_img.convert("L"))
    mask_arr[mask_arr > 0] = 255
    mask_img = Image.fromarray(mask_arr)
    mask_img = resize_and_crop(mask_img, size)
    mask_img = mask_processor.blur(mask_img, blur_factor=9)

    # Inference
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path, quality=95)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
