"""
Batch inference using fine-tuned CatVTON (LoRA weights)
========================================================
Supports single-GPU and multi-GPU inference via torch.multiprocessing.
Each GPU runs its own pipeline process on an independent slice of the pairs.

Usage (single GPU):
    python infer.py \
        --test_data    /workspace/dataset/test_data \
        --lora_weights /workspace/output/lora_mena/final \
        --output_dir   /workspace/test_results

Usage (all available GPUs):
    python infer.py \
        --test_data    /workspace/dataset/test_data \
        --lora_weights /workspace/output/lora_mena/final \
        --output_dir   /workspace/test_results \
        --num_gpus     -1

Usage (specific number of GPUs):
    python infer.py \
        --test_data    /workspace/dataset/test_data \
        --lora_weights /workspace/output/lora_mena/final \
        --output_dir   /workspace/test_results \
        --num_gpus     2
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageFilter
from tqdm import tqdm

from model.attn_processor import SkipAttnProcessor
from model.pipeline import CatVTONPipeline
from model.utils import init_adapter
from peft import PeftModel
from utils import resize_and_crop, resize_and_padding


# ── Argument parsing ──────────────────────────────────────────────────────────

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
        "--num_gpus",
        type=int,
        default=1,
        help=(
            "Number of GPUs to use. "
            "Set to -1 to use all available GPUs. "
            "Falls back to CPU if no GPUs are available."
        ),
    )
    p.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "unpaired"],
        help=(
            "Pairing mode when building pairs automatically (no test_pairs.txt). "
            "'paired'   → each person matched with their own garment. "
            "'unpaired' → each person matched with a different garment (shuffled by --seed)."
        ),
    )
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= "
            "1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the "
            "flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
        help="Whether or not to concatenate all conditions into one image.",
    )
    return p.parse_args()


# ── Pair building ─────────────────────────────────────────────────────────────

def build_paired(image_dir: Path, cloth_dir: Path) -> list[tuple[str, str]]:
    """Each person matched with their own garment (stem _0 -> _1)."""
    pairs = []
    for person_path in sorted(image_dir.iterdir()):
        if not person_path.stem.endswith("_0"):
            continue
        base = person_path.stem[:-2]
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = cloth_dir / f"{base}_1{ext}"
            if candidate.exists():
                pairs.append((person_path.name, candidate.name))
                break
    return pairs


def build_unpaired(image_dir: Path, cloth_dir: Path, seed: int) -> list[tuple[str, str]]:
    """Each person matched with a *different* person's garment (shuffled)."""
    valid = []
    for person_path in sorted(image_dir.iterdir()):
        if not person_path.stem.endswith("_0"):
            continue
        base = person_path.stem[:-2]
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = cloth_dir / f"{base}_1{ext}"
            if candidate.exists():
                valid.append((person_path, candidate))
                break

    if len(valid) < 2:
        raise ValueError("Need at least 2 valid person/garment pairs for unpaired mode.")

    person_paths = [p for p, _ in valid]
    cloth_paths  = [c for _, c in valid]

    rng = random.Random(seed)
    shuffled = cloth_paths.copy()
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(o.stem[:-2] != s.stem[:-2] for o, s in zip(cloth_paths, shuffled)):
            break
    else:
        shuffled = cloth_paths[1:] + cloth_paths[:1]

    return [(p.name, c.name) for p, c in zip(person_paths, shuffled)]


# ── Image helpers ─────────────────────────────────────────────────────────────

def do_repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np   = np.array(mask.convert("L")) / 255
    mask_np   = mask_np[:, :, None]  # (H, W) -> (H, W, 1) to broadcast over RGB
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    return Image.fromarray(repaint_result.astype(np.uint8))


def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


# ── Per-GPU worker ────────────────────────────────────────────────────────────

def run_worker(rank: int, args, pairs: list[tuple[str, str]], out_dir: Path):
    """
    Runs inference for a subset of pairs on a single GPU (or CPU).
    Each worker process owns its pipeline independently — no shared state.
    """
    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        device = "cpu"

    is_main = rank == 0
    if is_main:
        print(f"[GPU {rank}] Loading pipeline on {device} ...")

    # ── Load pipeline ─────────────────────────────────────────────────────────
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_ckpt,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version="mix",
        weight_dtype={
            "no":   torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device=device,
        skip_safety_check=True,
    )

    # ── Merge LoRA ────────────────────────────────────────────────────────────
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_weights)
    pipeline.unet = pipeline.unet.merge_and_unload()
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)
    pipeline.unet.eval()
    if is_main:
        print(f"[GPU {rank}] LoRA merged. Processing {len(pairs)} pairs ...")

    # ── Mask processor ────────────────────────────────────────────────────────
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    test_data = Path(args.test_data)
    image_dir = test_data / "image"
    cloth_dir = test_data / "cloth"
    mask_dir  = test_data / "agnostic-mask"
    size      = (args.width, args.height)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── Inference loop ────────────────────────────────────────────────────────
    # Only the main process shows the progress bar to avoid cluttered output
    pair_iter = tqdm(
        range(0, len(pairs), args.batch_size),
        desc=f"GPU {rank}",
        position=rank,
        disable=not is_main,
    )

    for batch_start in pair_iter:
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]

        person_imgs, cloth_imgs, mask_imgs, person_names = [], [], [], []
        for person_name, cloth_name in batch_pairs:
            person_img = Image.open(image_dir / person_name).convert("RGB")
            cloth_img  = Image.open(cloth_dir  / cloth_name).convert("RGB")

            person_img = resize_and_crop(person_img, size)
            cloth_img  = resize_and_padding(cloth_img, size)

            mask_fname = Path(person_name).stem + ".png"
            mask_path  = mask_dir / mask_fname
            if not mask_path.exists():
                mask_path = mask_dir / person_name
            mask_img = Image.open(mask_path)

            mask_arr = np.array(mask_img.convert("L"))
            mask_arr[mask_arr > 0] = 255
            mask_img = Image.fromarray(mask_arr)
            mask_img = resize_and_crop(mask_img, size)
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
            results.append(result[0] if isinstance(result, list) else result)

        # ── Save results ──────────────────────────────────────────────────────
        for i, result in enumerate(results):
            person_name = person_names[i]

            if args.repaint:
                mask_fname = Path(person_name).stem + ".png"
                mask_path  = mask_dir / mask_fname
                if not mask_path.exists():
                    mask_path = mask_dir / person_name
                original_person = Image.open(image_dir / person_name).convert("RGB").resize(result.size, Image.LANCZOS)
                original_mask   = Image.open(mask_path).resize(result.size, Image.NEAREST)
                result = do_repaint(original_person, original_mask, result)

            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new("RGB", (w * 3, h))
                concated_result.paste(person_imgs[i], (0, 0))
                concated_result.paste(cloth_imgs[i], (w, 0))
                concated_result.paste(result, (w * 2, 0))
                result = concated_result

            stem = Path(person_name).stem
            result.save(out_dir / f"{stem}_result.jpg", quality=95)

    if is_main:
        print(f"[GPU {rank}] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    test_data = Path(args.test_data)
    image_dir = test_data / "image"
    cloth_dir = test_data / "cloth"

    # ── Build pairs once in the main process ──────────────────────────────────
    pairs_file = test_data / "test_pairs.txt"
    if pairs_file.exists():
        pairs = []
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        print(f"Loaded {len(pairs)} pairs from test_pairs.txt")
    else:
        if args.mode == "paired":
            pairs = build_paired(image_dir, cloth_dir)
        else:
            pairs = build_unpaired(image_dir, cloth_dir, seed=args.seed)
        print(f"Auto-built {len(pairs)} {args.mode} pairs")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Decide how many GPUs to use ───────────────────────────────────────────
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs found — running on CPU.")
        num_gpus = 1
    elif args.num_gpus == -1:
        num_gpus = available_gpus
    else:
        num_gpus = min(args.num_gpus, available_gpus)
        if num_gpus < args.num_gpus:
            print(f"Warning: requested {args.num_gpus} GPUs but only {available_gpus} available. Using {num_gpus}.")

    print(f"Using {num_gpus} GPU(s) across {len(pairs)} pairs.")

    # ── Split pairs evenly across workers (round-robin for equal load) ────────
    chunks = [pairs[i::num_gpus] for i in range(num_gpus)]

    if num_gpus == 1:
        # No multiprocessing overhead for single GPU / CPU
        run_worker(0, args, chunks[0], out_dir)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=run_worker,
                args=(rank, args, chunks[rank], out_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Surface any worker failures
        for rank, p in enumerate(processes):
            if p.exitcode != 0:
                raise RuntimeError(f"Worker on GPU {rank} exited with code {p.exitcode}.")

    print(f"\nDone! {len(pairs)} results saved to: {out_dir}")


if __name__ == "__main__":
    main()
