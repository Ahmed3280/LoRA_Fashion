import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image

from model.attn_processor import SkipAttnProcessor
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from model.utils import get_trainable_module, init_adapter
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    parser = argparse.ArgumentParser(
        description="CatVTON Gradio demo with LoRA fine-tuned weights."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help="Path or Hub ID of the base SD-inpainting model.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="Path or Hub ID of the CatVTON attention checkpoint.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Local path to the directory containing the saved LoRA weights "
             "(the folder produced by train_lora.py, e.g. ./output/lora_mena/final).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output_lora",
        help="Directory where result images are written.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width to resize input images to before inference.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height to resize input images to before inference.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for inference.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Allow TF32 on Ampere GPUs for faster inference.",
    )
    parser.add_argument(
        "--repaint",
        action="store_true",
        help="Blend result back onto the original person background.",
    )
    return parser.parse_args()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
args = parse_args()

# Download / locate the CatVTON attention checkpoint
repo_path = snapshot_download(repo_id=args.resume_path)

# Build pipeline (loads base UNet + CatVTON attention weights)
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device="cuda",
    skip_safety_check=True,
)

# Inject LoRA adapter into the pipeline's UNet
print(f"Loading LoRA weights from: {args.lora_path}")
pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_path)
pipeline.unet = pipeline.unet.merge_and_unload()   # fuse LoRA into weights for faster inference
pipeline.unet.to("cuda", dtype=init_weight_dtype(args.mixed_precision))
print("LoRA weights loaded and merged.")

# Mask processor + AutoMasker (for automatic masking fallback)
mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------
def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type,
):
    # person_image is a dict from gr.ImageEditor: background + painted layers
    person_path = person_image["background"]
    mask_layer  = person_image["layers"][0]

    # Parse drawn mask (if any)
    mask = Image.open(mask_layer).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        # No mask drawn → fall back to AutoMasker
        mask = None
    else:
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 0] = 255
        mask = Image.fromarray(mask_arr)

    # Prepare output path
    os.makedirs(args.output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    day_dir = os.path.join(args.output_dir, date_str[:8])
    os.makedirs(day_dir, exist_ok=True)
    result_save_path = os.path.join(day_dir, date_str[8:] + ".png")

    # Seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load & resize images
    person_image = Image.open(person_path).convert("RGB")
    cloth_image  = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image  = resize_and_padding(cloth_image, (args.width, args.height))

    # Resolve mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(person_image, cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    # Run pipeline
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )[0]

    # Save grid (person | masked | cloth | result)
    masked_person = vis_mask(person_image, mask)
    image_grid([person_image, masked_person, cloth_image, result_image], 1, 4).save(
        result_save_path
    )

    # Build display output
    if show_type == "result only":
        return result_image

    width, height = person_image.size
    if show_type == "input & result":
        condition_width = width // 2
        conditions = image_grid([person_image, cloth_image], 2, 1)
    else:  # "input & mask & result"
        condition_width = width // 3
        conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)

    conditions = conditions.resize((condition_width, height), Image.NEAREST)
    composite = Image.new("RGB", (width + condition_width + 5, height))
    composite.paste(conditions, (0, 0))
    composite.paste(result_image, (condition_width + 5, 0))
    return composite


def person_example_fn(image_path):
    return image_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
HEADER = f"""
<h1 style="text-align: center;">CatVTON – LoRA Fine-Tuned Inference</h1>
<p style="text-align: center;">
  Testing fine-tuned LoRA weights &nbsp;|&nbsp;
  Resolution: <b>{args.width} × {args.height}</b> &nbsp;|&nbsp;
  LoRA path: <code>{args.lora_path}</code>
</p>
<p style="text-align: center; color: #808080;">
  Draw a mask on the person image to override auto-masking, or select a cloth type for automatic masking.
</p>
"""


def app_gradio():
    with gr.Blocks(title="CatVTON LoRA") as demo:
        gr.HTML(HEADER)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image = gr.ImageEditor(
                        interactive=True,
                        label="Person Image  (draw mask with 🖌️ to override auto-mask)",
                        type="filepath",
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True,
                            label="Garment / Condition Image",
                            type="filepath",
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            "<span style='color:#808080;font-size:small;'>"
                            "Mask priority:<br>"
                            "1. Drawn mask (higher priority)<br>"
                            "2. Auto-mask from cloth type"
                            "</span>"
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )

                submit = gr.Button("Run Try-On", variant="primary")
                gr.Markdown(
                    "<center><span style='color:#FF0000'>Click once and wait — inference may take ~30 s on GPU.</span></center>"
                )

                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=50,
                    )
                    guidance_scale = gr.Slider(
                        label="CFG Strength",
                        minimum=0.0,
                        maximum=7.5,
                        step=0.5,
                        value=2.5,
                    )
                    seed = gr.Slider(
                        label="Seed  (-1 = random)",
                        minimum=-1,
                        maximum=10000,
                        step=1,
                        value=42,
                    )
                    show_type = gr.Radio(
                        label="Display Mode",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            # ── Right column: output ─────────────────────────────────────
            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")

        # Wire events
        image_path.change(person_example_fn, inputs=image_path, outputs=person_image)

        submit.click(
            submit_function,
            inputs=[
                person_image,
                cloth_image,
                cloth_type,
                num_inference_steps,
                guidance_scale,
                seed,
                show_type,
            ],
            outputs=result_image,
        )

    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
