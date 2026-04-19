"""Standalone CatVTON (optional LoRA) single-image try-on.

Usable as a CLI:
    python infer_cli.py --person P.jpg --mask M.png --garment G.jpg \
        --lora_path epoch_0010 --output out.png

Or as a library:
    from infer_cli import load_pipeline, run_inference
    pipe = load_pipeline(lora_path="epoch_0010")
    result = run_inference(pipe, person_pil, mask_pil, garment_pil)

Mirrors app_lora.py's pipeline construction, LoRA merge, preprocessing,
and inference call exactly. No dependency on app_lora.py.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

logger = logging.getLogger(__name__)


def load_pipeline(
    lora_path: Optional[str] = None,
    base_model_path: str = "booksforcharlie/stable-diffusion-inpainting",
    resume_path: str = "zhengchong/CatVTON",
    width: int = 768,
    height: int = 1024,
    precision: str = "bf16",
    allow_tf32: bool = True,
) -> Dict[str, Any]:
    """Build CatVTONPipeline, optionally merge a LoRA adapter, and return a reusable handle.

    Call ONCE at worker startup. Safe to reuse forever.
    """
    # Download / locate the CatVTON attention checkpoint
    repo_path = snapshot_download(repo_id=resume_path)

    # Build pipeline (loads base UNet + CatVTON attention weights)
    pipeline = CatVTONPipeline(
        base_ckpt=base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(precision),
        use_tf32=allow_tf32,
        device="cuda",
        skip_safety_check=True,
    )

    if lora_path is not None:
        # Inject LoRA adapter into the pipeline's UNet
        logger.info("Loading LoRA weights from: %s", lora_path)
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
        pipeline.unet = pipeline.unet.merge_and_unload()   # fuse LoRA into weights for faster inference
        pipeline.unet.to("cuda", dtype=init_weight_dtype(precision))
        logger.info("LoRA weights loaded and merged.")
    else:
        logger.info("No LoRA path provided; running vanilla CatVTON.")

    # Mask processor (for blur-before-inference, matches app_lora.py)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    return {
        "pipeline": pipeline,
        "mask_processor": mask_processor,
        "width": width,
        "height": height,
        "repo_path": repo_path,
    }


def run_inference(
    pipeline: Dict[str, Any],
    person_image: Image.Image,
    mask_image: Image.Image,
    garment_image: Image.Image,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    seed: int = 42,
) -> Image.Image:
    """Run a single try-on inference on 3 PIL images and return a PIL result.

    Args:
        pipeline: dict returned by load_pipeline.
        person_image: PIL image of the person.
        mask_image: PIL mask (agnostic region) — caller provides; no AutoMasker.
        garment_image: PIL image of the garment.
        num_inference_steps: Scheduler steps.
        guidance_scale: Classifier-free guidance.
        seed: int RNG seed; pass -1 for truly random (generator=None).

    Returns:
        PIL.Image.Image at (width, height).
    """
    pipe = pipeline["pipeline"]
    mask_processor: VaeImageProcessor = pipeline["mask_processor"]
    width = pipeline["width"]
    height = pipeline["height"]

    # Seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load & resize images
    person_image = person_image.convert("RGB")
    cloth_image  = garment_image.convert("RGB")
    person_image = resize_and_crop(person_image, (width, height))
    cloth_image  = resize_and_padding(cloth_image, (width, height))

    # Resolve mask (caller always provides one here)
    mask = resize_and_crop(mask_image.convert("L"), (width, height))
    mask = mask_processor.blur(mask, blur_factor=9)

    # Run pipeline
    result_image = pipe(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )[0]

    return result_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatVTON single-image try-on (optional LoRA).")
    parser.add_argument("--person", type=str, required=True, help="Path to person image.")
    parser.add_argument("--mask", type=str, required=True, help="Path to agnostic mask image.")
    parser.add_argument("--garment", type=str, required=True, help="Path to garment image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save result PNG.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter directory. If omitted, runs vanilla CatVTON.")
    parser.add_argument("--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--resume_path", type=str, default="zhengchong/CatVTON",
                        help="CatVTON attention ckpt (HF repo id or local path, e.g. a mounted volume).")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    pipeline = load_pipeline(
        lora_path=args.lora_path,
        base_model_path=args.base_model_path,
        resume_path=args.resume_path,
        width=args.width,
        height=args.height,
        precision=args.mixed_precision,
    )

    person = Image.open(args.person)
    mask = Image.open(args.mask)
    garment = Image.open(args.garment)

    result = run_inference(
        pipeline,
        person_image=person,
        mask_image=mask,
        garment_image=garment,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    result.save(args.output)
    logger.info("Saved result to %s (size=%s)", args.output, result.size)


if __name__ == "__main__":
    main()
