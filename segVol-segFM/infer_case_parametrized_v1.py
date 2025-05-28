#!/usr/bin/env python
"""SegVol single–organ inference script

Example usage:
    python infer_case_parametrized_v1.py \
        --data_glob "/path/to/*.npz" \
        --model_dir ./segvol \
        --ckpt ./ckpts_mobilenet_2_5d/epoch_100_loss_0.2663_mobilenet_2_5d.pth \
        --out_dir ./outputs_epoch_100_loss_0.2663_mobilenet_2_5d_v2 \
        --device cuda:0

python infer_case_parametrized_v1.py \
    --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
    --model_dir ./segvol \
    --ckpt ./ckpts_mobilenet_2_5d/epoch_100_loss_0.2663_mobilenet_2_5d.pth \
    --out_dir ./outputs_epoch_100_loss_0.2663_mobilenet_2_5d_v2 \
    --device cuda:0

The script replicates the original notebook-style code but exposes all hard‑coded
paths as command‑line flags so it can be dropped into a pipeline or cluster
job scheduler with different data / checkpoints.
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

from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel  # noqa: E402 – local import

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser("SegVol inference")
    parser.add_argument("--data_glob", required=True, type=str,
                        help="Glob expression that expands to *.npz cases to segment")
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory containing SegVol model files (config + tokenizer)")
    parser.add_argument("--ckpt", required=True, type=str,
                        help="Path to the .pth checkpoint to load weights from")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Directory where compressed .npz predictions will be written")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Torch device specifier, e.g. cuda:0 or cpu (default: cuda:0)")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Loading utilities
# --------------------------------------------------------------------------------------

def load_item(npz_file: str | Path, preprocessor):
    """Load an individual *.npz case, run preprocessing and cube creation.

    If the case does not include a `boxes` key, the entire volume is used as a
    single bounding‑box prompt and a flag is returned so the caller can count
    such cases.
    """
    npz_content = np.load(npz_file, allow_pickle=True)
    imgs = npz_content["imgs"].astype(np.float32)

    # --- preprocess CT volume --------------------------------------------------------
    imgs = preprocessor.preprocess_ct_case(imgs)  # -> (1, H, W, D)

    boxes_missing = False
    if "boxes" in npz_content:
        boxes = npz_content["boxes"]  # list[list[int, ...]]
    else:
        boxes_missing = True
        _, H, W, D = imgs.shape
        boxes = [[0, 0, 0, H, W, D]]  # cover whole volume

    # --- convert boxes to binary cubes ---------------------------------------------
    cube_boxes = []
    for std_box in boxes:
        # Accept either dict or list[6]
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

    cube_boxes = torch.stack(cube_boxes, dim=0)  # (N, H, W, D)
    assert cube_boxes.shape[1:] == imgs.shape[1:], (
        f"Cube shape {cube_boxes.shape[1:]} != CT shape {imgs.shape[1:]}"
    )

    zoom_item = preprocessor.zoom_transform_case(imgs, cube_boxes)
    zoom_item["file_path"] = str(npz_file)
    zoom_item["img_original"] = torch.from_numpy(npz_content["imgs"]).float()
    zoom_item["cube_boxes"] = cube_boxes
    zoom_item["boxes_missing"] = boxes_missing
    return zoom_item


def backfill_foreground_preds(
    ct_shape: tuple[int, ...],
    logits_mask: torch.Tensor,
    start_coord: torch.Tensor,
    end_coord: torch.Tensor,
):
    """Insert the zoomed‑in prediction back into the original volume shape."""
    binary_preds = torch.zeros(ct_shape, dtype=torch.float32, device=logits_mask.device)
    binary_preds[
        start_coord[0] : end_coord[0],
        start_coord[1] : end_coord[1],
        start_coord[2] : end_coord[2],
    ] = torch.sigmoid(logits_mask)
    return (binary_preds > 0.5).float()


def infer_case(model_val: SegVolModel, data_item, processor, device: torch.device):
    """Run inference for a single CT case."""
    data_item["image"] = data_item["image"].unsqueeze(0).to(device)
    data_item["zoom_out_image"] = data_item["zoom_out_image"].unsqueeze(0).to(device)

    start_coord = data_item["foreground_start_coord"]
    end_coord = data_item["foreground_end_coord"]
    img_original = data_item["img_original"]

    category_n = data_item["cube_boxes"].shape[0]
    final_preds = torch.zeros_like(img_original, device=device)

    for cls_idx in range(category_n):
        cube_boxes = data_item["cube_boxes"][cls_idx].unsqueeze(0).unsqueeze(0).to(device)
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

        # back‑fill into original CT space
        binary_preds = backfill_foreground_preds(
            img_original.shape, logits_mask, start_coord, end_coord
        )
        final_preds[binary_preds == 1] = cls_idx + 1
        torch.cuda.empty_cache()

    return final_preds.cpu().numpy().astype(np.uint16)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ---- device -------------------------------------------------------------------
    try:
        device = torch.device(args.device)
    except RuntimeError:
        device = torch.device("cpu")
        print(f"[Warning] Requested device {args.device!r} unavailable – falling back to CPU.")

    # ---- model + processor ---------------------------------------------------------
    clip_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer

    checkpoint = torch.load(args.ckpt, map_location=device)
    model_val.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model_val.eval().to(device)
    processor = model_val.processor

    # ---- I/O setup -----------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(glob(args.data_glob))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files matched the pattern {args.data_glob!r}. Check the path.")

    missing_boxes_count = 0

    # ---- batch processing ----------------------------------------------------------
    with tqdm(total=len(npz_files), desc="Inferring") as pbar:
        for npz_file in npz_files:
            data_item = load_item(npz_file, processor)
            if data_item.get("boxes_missing", False):
                missing_boxes_count += 1

            preds = infer_case(model_val, data_item, processor, device)

            out_path = out_dir / (Path(npz_file).stem + "_seg.npz")
            np.savez_compressed(out_path, segs=preds)
            pbar.update(1)

    print("Inference complete ✅")
    print(f"Total cases processed : {len(npz_files)}")
    print(f"Cases without 'boxes' : {missing_boxes_count}")


if __name__ == "__main__":
    main()
