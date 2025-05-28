#!/usr/bin/env python
"""SegVol single–organ inference script (parameterised)

This version mirrors the behaviour of the original notebook-style code **line‑for‑line**
while exposing the previously hard‑coded paths as command‑line flags. Output file
names, tensor devices, logging, and thresholding logic are identical to the
source you provided – the only changes are the parameters.

Example:
python infer_case_parametrized_fast.py \
        --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_100_loss_0.2663_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_100_loss_0.2663_mobilenet_2_5d.pth \
        --device cuda:0

python infer_case_parametrized_fast.py \
        --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_150_loss_0.2532_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_150_loss_0.2532_mobilenet_2_5d.pth \
        --device cuda:0

    
python infer_case_parametrized_fast.py \
        --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_150_loss_0.2532_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_150_loss_0.2532_mobilenet_2_5d.pth \
        --device cuda:0

python infer_case_parametrized_fast.py \
        --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_175_loss_0.2454_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_175_loss_0.2454_mobilenet_2_5d.pth \
        --device cuda:1

python infer_case_parametrized_fast.py \
        --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_200_loss_0.2428_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_200_loss_0.2428_mobilenet_2_5d.pth \
        --device cuda:1
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel  # noqa: E402

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SegVol inference (parameterised)")
    p.add_argument("--data_glob", required=True, type=str,
                   help="Glob expression matching *.npz cases to segment")
    p.add_argument("--model_dir", required=True, type=str,
                   help="Directory with SegVol config + tokenizer")
    p.add_argument("--ckpt", required=True, type=str,
                   help="Path to .pth checkpoint")
    p.add_argument("--out_dir", required=True, type=str,
                   help="Directory for *.npz predictions (identical base names)")
    p.add_argument("--device", default="cuda:0", type=str,
                   help="Torch device spec (cuda:0 | cpu). Default: cuda:0")
    return p.parse_args()

# --------------------------------------------------------------------------------------
# Helpers (unchanged logic)
# --------------------------------------------------------------------------------------

def load_item(npz_file: str | Path, preprocessor):
    file_path = str(npz_file)
    npz = np.load(file_path, allow_pickle=True)
    imgs = npz["imgs"]
    print("Keys:", list(npz.keys()))

    # preprocessing exactly as before
    imgs = preprocessor.preprocess_ct_case(imgs)  # -> (1, H, W, D)

    if "boxes" in npz:
        boxes = npz["boxes"]
    else:
        print(f"Warning: 'boxes' missing in {file_path}, using entire image as default")
        _, H, W, D = imgs.shape
        boxes = [[0, 0, 0, H, W, D]]

    cube_boxes = []
    for std_box in boxes:
        if isinstance(std_box, (list, tuple)) and len(std_box) == 6:
            std_box = {
                "z_min": std_box[0],
                "z_mid_y_min": std_box[1],
                "z_mid_x_min": std_box[2],
                "z_max": std_box[3],
                "z_mid_y_max": std_box[4],
                "z_mid_x_max": std_box[5],
            }
        cube_boxes.append(build_binary_cube_dict(std_box, imgs.shape[1:]))
    cube_boxes = torch.stack(cube_boxes, dim=0)
    assert cube_boxes.shape[1:] == imgs.shape[1:], f"{cube_boxes.shape} != {imgs.shape}"

    zoom_item = preprocessor.zoom_transform_case(imgs, cube_boxes)
    zoom_item["file_path"] = file_path
    zoom_item["img_original"] = torch.from_numpy(npz["imgs"])
    # Note: the original script relied on preprocessor.zoom_transform_case to
    # populate 'cube_boxes' & 'zoom_out_cube_boxes'. We leave that untouched.
    return zoom_item


def backfill_foreground_preds(ct_shape, logits_mask, start_coord, end_coord):
    binary_preds = torch.zeros(ct_shape)
    binary_preds[
        start_coord[0] : end_coord[0],
        start_coord[1] : end_coord[1],
        start_coord[2] : end_coord[2],
    ] = torch.sigmoid(logits_mask)
    binary_preds = torch.where(binary_preds > 0.5, 1.0, 0.0)
    return binary_preds


def infer_case(model_val, data_item, processor, device):
    data_item["image"], data_item["zoom_out_image"] = (
        data_item["image"].unsqueeze(0).to(device),
        data_item["zoom_out_image"].unsqueeze(0).to(device),
    )
    start_coord, end_coord = (
        data_item["foreground_start_coord"],
        data_item["foreground_end_coord"],
    )

    img_original = data_item["img_original"]
    category_n = data_item["cube_boxes"].shape[0]
    category_ids = list(torch.arange(category_n) + 1)
    final_preds = torch.zeros_like(img_original)

    for category_id in category_ids:
        cls_idx = (category_id - 1).item()
        cube_boxes = data_item["cube_boxes"][cls_idx].unsqueeze(0).unsqueeze(0)
        bbox_prompt = processor.bbox_prompt_b(
            data_item["zoom_out_cube_boxes"][cls_idx], device=device
        )
        with torch.no_grad():
            logits_mask = model_val.forward_test(
                image=data_item["image"],
                zoomed_image=data_item["zoom_out_image"],
                bbox_prompt_group=[bbox_prompt, cube_boxes],
                use_zoom=True,
            )
        binary_preds = backfill_foreground_preds(
            img_original.shape, logits_mask, start_coord, end_coord
        )
        final_preds[binary_preds == 1] = category_id
        torch.cuda.empty_cache()

    return final_preds.numpy()

# --------------------------------------------------------------------------------------
# Main (param‑driven, behaviour‑identical)
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # env var as in original
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    # model + processor setup (unchanged)
    clip_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor

    checkpoint = torch.load(args.ckpt, map_location=device)
    model_val.load_state_dict(checkpoint["model_state_dict"])
    model_val.eval().to(device)

    # I/O
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = glob(args.data_glob)
    missing_boxes_count = 0

    with tqdm(total=len(npz_files), desc="Processing files") as pbar:
        for npz_file in npz_files:
            data_item = load_item(npz_file, processor)
            if "Warning: 'boxes' missing" in data_item.get("warnings", ""):
                missing_boxes_count += 1
            final_preds = infer_case(model_val, data_item, processor, device)
            output_path = out_dir / Path(npz_file).name  # **same base name as input**
            np.savez_compressed(output_path, segs=final_preds)
            pbar.update(1)

    print(f"Done. Total files processed: {len(npz_files)}")
    print(f"Total files missing \"boxes\": {missing_boxes_count}")


if __name__ == "__main__":
    main()
