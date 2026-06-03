
#!/usr/bin/env python3

import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor


SAM_CKPT = "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MIN_REGION_PX = 10
SPECK_SIZE = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Garment mapping from your API format
# -------------------------------------------------------------------

SLUG_TO_CLOTH_TYPE = {
    # Upper body
    "blazer": "upper",
    "cardigan": "upper",
    "formal-shirt": "upper",
    "hoodie": "upper",
    "jacket": "upper",
    "polo-shirt": "upper",
    "sweatshirt": "upper",
    "tank-top": "upper",
    "t-shirt": "upper",
    "vest": "upper",
    "coat": "upper",

    # Lower body
    "cargo-pants": "lower",
    "jeans": "lower",
    "joggers": "lower",
    "shorts": "lower",
    "skirt": "lower",
    "sweatpants": "lower",

    # Overall / full body
    "abaya": "overall",
    "dress": "overall",
    "jalabiya": "overall",
    "kaftan": "overall",
    "thobe": "overall",
}

GARMENT_CLASSES = {
    "upper": [
        "shirt",
        "t-shirt",
        "blouse",
        "hoodie",
        "sweater",
        "jacket",
        "coat",
        "blazer",
        "cardigan",
        "tunic",
    ],

    "lower": [
        "pants",
        "jeans",
        "trousers",
        "skirt",
        "shorts",
        "leggings",
    ],

    "overall": [
        "abaya",
        "dress",
        "gown",
        "robe",
        "kaftan",
        "thobe",
        "kandura",
        "jalabiya",
        "jumpsuit",
    ],
}

HIJAB_CLASSES = [
    "hijab",
    "niqab",
    "khimar",
    "headscarf",
    "keffiyeh",
    "ghutrah",
    "agal",
]

NON_CLOTHING = [
    "face", "hands", "hair", "neck", "forehead", "feet", "skin", "head",
    "sunglasses", "glasses", "eyeglasses",
    "beard", "mustache", "facial hair",
    "shoes", "sandals", "slippers", "sneakers", "boots",
    "phone", "earring", "ring", "necklace", "bracelet", "watch",
    "bag", "purse", "handbag",
    "background",
]


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def ensure_sam_checkpoint():
    if os.path.exists(SAM_CKPT):
        return

    print("Downloading SAM checkpoint...")
    urllib.request.urlretrieve(SAM_URL, SAM_CKPT)
    print("Download complete.")


def matches_any(phrase, class_list):
    import re

    def tokenise(s):
        return set(re.split(r"[\s\-_/]+", s.lower().strip()))

    phrase_tokens = tokenise(phrase)

    for cls in class_list:
        cls_tokens = tokenise(cls)

        shorter, longer = (
            (phrase_tokens, cls_tokens)
            if len(phrase_tokens) <= len(cls_tokens)
            else (cls_tokens, phrase_tokens)
        )

        if shorter and shorter.issubset(longer):
            return True

    return False


def predict_best_masks(predictor, boxes_xyxy, image_shape, device):
    best_masks = []

    for box in boxes_xyxy:
        box_t = predictor.transform.apply_boxes_torch(
            box.unsqueeze(0),
            image_shape
        ).to(device)

        preds, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_t,
            multimask_output=True,
        )

        best_idx = scores[0].argmax().item()
        best_masks.append(preds[0, best_idx].cpu().numpy())

    if best_masks:
        return np.stack(best_masks)

    return np.zeros((0, *image_shape[:2]), dtype=bool)


def build_group_mask(indices, masks_np):
    out = np.zeros(masks_np.shape[1:], dtype=np.uint8)

    for i in indices:
        out[masks_np[i]] = 255

    return out


def clean_mask(mask_uint8, min_region_size=500):
    filled = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)),
        iterations=1,
    )

    opened = cv2.morphologyEx(
        filled,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened,
        connectivity=8
    )

    out = np.zeros_like(opened)

    for idx in range(1, n_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_region_size:
            out[labels == idx] = 255

    return out


def remove_specks(mask_uint8, speck_size=50):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8
    )

    out = mask_uint8.copy()

    for idx in range(1, n_labels):
        if stats[idx, cv2.CC_STAT_AREA] < speck_size:
            out[labels == idx] = 0

    return out


def build_text_prompt(cloth_type):
    all_classes = (
        HIJAB_CLASSES +
        GARMENT_CLASSES[cloth_type] +
        NON_CLOTHING
    )

    return ". ".join(all_classes) + "."


# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------

ensure_sam_checkpoint()

print("Loading GroundingDINO...")
processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(device)

print("Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device)
predictor = SamPredictor(sam)

print(f"Using device: {device}")


# -------------------------------------------------------------------
# Main mask generation
# -------------------------------------------------------------------

def generate_mask(image_path, cloth_type, output_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    H, W = image_np.shape[:2]

    text_prompt = build_text_prompt(cloth_type)

    inputs = processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gdino(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(H, W)],
    )[0]

    boxes = results["boxes"].cpu()
    phrases = list(results["labels"])

    print(f"Detected regions: {phrases}")

    if len(phrases) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        Image.fromarray(empty).save(output_path)
        return

    predictor.set_image(image_np)

    masks = predict_best_masks(
        predictor,
        boxes,
        image_np.shape[:2],
        device
    )

    hijab_idx = []
    clothing_idx = []
    exclude_idx = []

    clothing_classes = GARMENT_CLASSES[cloth_type]

    for i, phrase in enumerate(phrases):
        if matches_any(phrase, HIJAB_CLASSES):
            hijab_idx.append(i)

        elif matches_any(phrase, clothing_classes):
            clothing_idx.append(i)

        elif matches_any(phrase, NON_CLOTHING):
            exclude_idx.append(i)

    raw_clothing = build_group_mask(clothing_idx, masks)
    raw_hijab = build_group_mask(hijab_idx, masks)
    raw_exclusion = build_group_mask(exclude_idx, masks)

    raw_clothing[raw_exclusion == 255] = 0
    raw_clothing[raw_hijab == 255] = 0

    FACE_LABELS = {
        "face", "hair", "neck", "forehead", "skin",
        "beard", "mustache", "facial hair",
        "sunglasses", "glasses", "eyeglasses",
    }

    LIMB_LABELS = {
        "hand", "hands", "arm", "forearm",
        "wrist", "feet", "foot"
    }

    ACCESSORY_LABELS = {
        "ring", "earring", "necklace", "bracelet", "watch",
        "bag", "purse", "handbag", "phone",
        "shoes", "sandals", "slippers", "sneakers", "boots",
    }

    raw_face_excl = np.zeros(image_np.shape[:2], dtype=np.uint8)
    raw_limb_excl = np.zeros(image_np.shape[:2], dtype=np.uint8)
    raw_acc_excl = np.zeros(image_np.shape[:2], dtype=np.uint8)

    for i, phrase in enumerate(phrases):
        p = phrase.lower()

        if any(lbl in p or p in lbl for lbl in FACE_LABELS):
            raw_face_excl[masks[i]] = 255

        elif any(lbl in p or p in lbl for lbl in LIMB_LABELS):
            raw_limb_excl[masks[i]] = 255

        elif any(lbl in p or p in lbl for lbl in ACCESSORY_LABELS):
            raw_acc_excl[masks[i]] = 255

    face_excl_dilated = cv2.dilate(
        raw_face_excl,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    limb_excl_dilated = cv2.dilate(
        raw_limb_excl,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    def finalize(raw):
        m = clean_mask(raw, min_region_size=MIN_REGION_PX)

        m[face_excl_dilated == 255] = 0
        m[limb_excl_dilated == 255] = 0
        m[raw_acc_excl == 255] = 0

        m = remove_specks(m, speck_size=SPECK_SIZE)

        return m

    final_mask = finalize(raw_clothing)

    Image.fromarray(final_mask).save(output_path)

    print(f"Saved mask to: {output_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to person image"
    )

    parser.add_argument(
        "--garment-json",
        required=True,
        help="JSON object or JSON file containing garment data"
    )

    parser.add_argument(
        "--output",
        default="mask.png",
        help="Output mask path"
    )

    args = parser.parse_args()

    garment_input = args.garment_json

    if os.path.isfile(garment_input):
        with open(garment_input, "r", encoding="utf-8") as f:
            garment_data = json.load(f)
    else:
        garment_data = json.loads(garment_input)

    slug = garment_data["slug"]

    if slug not in SLUG_TO_CLOTH_TYPE:
        raise ValueError(
            f"Unsupported garment slug: {slug}\n"
            f"Supported slugs: {sorted(SLUG_TO_CLOTH_TYPE.keys())}"
        )

    cloth_type = SLUG_TO_CLOTH_TYPE[slug]

    print(f"Garment slug: {slug}")
    print(f"Mapped cloth type: {cloth_type}")

    generate_mask(
        image_path=args.image,
        cloth_type=cloth_type,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
