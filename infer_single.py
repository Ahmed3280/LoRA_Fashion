"""
Single-image inference for fine-tuned CatVTON (LoRA weights)
=============================================================
Change the 5 variables in the CONFIG block below, then run:
    python infer_single.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← only edit this block
# ─────────────────────────────────────────────────────────────────────────────

# 1. Base SD-inpainting checkpoint (HuggingFace repo ID or local folder)
BASE_CKPT = "booksforcharlie/stable-diffusion-inpainting"

# 2. CatVTON attention weights (HuggingFace repo ID or local folder)
ATTN_CKPT = "zhengchong/CatVTON"

# 3. Fine-tuned LoRA weights folder produced by train_lora.py
LORA_WEIGHTS = "./output/lora_mena/final"

# 4. Input paths
PERSON_IMAGE = "./inputs/person.jpg"
MASK_IMAGE   = "./inputs/mask.png"
CLOTH_IMAGE  = "./inputs/cloth.jpg"

# ─────────────────────────────────────────────────────────────────────────────
# (optional tweaks — rarely need to change)
OUTPUT_PATH   = "./output_single.jpg"
HEIGHT, WIDTH = 1024, 768
STEPS         = 50
GUIDANCE      = 2.5
SEED          = 42
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from model.attn_processor import SkipAttnProcessor
from model.pipeline import CatVTONPipeline
from model.utils import init_adapter
from peft import PeftModel
from utils import resize_and_crop, resize_and_padding


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(SEED)

    # ── Load pipeline ──────────────────────────────────────────────────────
    print("Loading base pipeline ...")
    pipeline = CatVTONPipeline(
        base_ckpt=BASE_CKPT,
        attn_ckpt=ATTN_CKPT,
        attn_ckpt_version="mix",
        weight_dtype=torch.bfloat16,
        device=device,
        skip_safety_check=True,
    )

    # ── Merge LoRA weights ─────────────────────────────────────────────────
    print(f"Loading LoRA weights from: {LORA_WEIGHTS}")
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, LORA_WEIGHTS)
    pipeline.unet = pipeline.unet.merge_and_unload()
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)
    pipeline.unet.eval()
    print("LoRA merged.")

    # ── Mask processor (blur for smooth edges) ────────────────────────────
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    size = (WIDTH, HEIGHT)

    # ── Load & preprocess inputs ──────────────────────────────────────────
    print("Preprocessing inputs ...")
    person_img = resize_and_crop(Image.open(PERSON_IMAGE).convert("RGB"), size)
    cloth_img  = resize_and_padding(Image.open(CLOTH_IMAGE).convert("RGB"), size)

    mask_arr = np.array(Image.open(MASK_IMAGE).convert("L"))
    mask_arr[mask_arr > 0] = 255
    mask_img = resize_and_crop(Image.fromarray(mask_arr), size)
    mask_img = mask_processor.blur(mask_img, blur_factor=9)

    # ── Run inference ──────────────────────────────────────────────────────
    print("Running inference ...")
    result = pipeline(
        image=person_img,
        condition_image=cloth_img,
        mask=mask_img,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )[0]

    result.save(OUTPUT_PATH, quality=95)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
