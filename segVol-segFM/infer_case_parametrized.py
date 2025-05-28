#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametrised inference script (prediction-only).

• Handles missing `boxes` by using the screenshot logic:
  iterate over the `prompt` dict (if it exists) and, for every
  entry but `'instance_label'`, create one full-volume box.

• Produces one .npz per input containing `segs`.

python infer_case_parametrized.py \
    --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
    --model_dir ./segvol \
    --ckpt ./ckpts_mobilenet_2_5d/epoch_100_loss_0.2663_mobilenet_2_5d.pth \
    --out_dir ./outputs_epoch_100_loss_0.2663_mobilenet_2_5d \
    --device cuda:0


"""

from __future__ import annotations
import os, json, argparse
from glob import glob
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel

# --------------------------------------------------------------------------- #
# ----------------------------- helper functions ---------------------------- #
# --------------------------------------------------------------------------- #
def make_full_volume_box(shape_3d: tuple[int, int, int]) -> List[int]:
    """Return [z_min, y_min, x_min, z_max, y_max, x_max] covering entire CT."""
    _, H, W, D = shape_3d  # (1,H,W,D)
    return [0, 0, 0, H, W, D]


def boxes_from_prompt(
    imgs: np.ndarray,
    prompt: Dict[str, Any] | None,
) -> List[List[int]]:
    """
    Generate a list of boxes according to screenshot logic:

    for each key in `prompt` (sorted), **except** 'instance_label',
    push one whole-volume box into the list.  If `prompt` is None or
    contains no numeric keys, fall back to a single full-volume box.
    """
    full_box = make_full_volume_box(imgs.shape)
    if prompt is None:
        return [full_box]

    # filter numeric prompt IDs and keep order deterministic
    numeric_items = [
        (pid, txt) for pid, txt in sorted(prompt.items()) if pid.isdigit()
    ]

    if not numeric_items:
        return [full_box]

    boxes: List[List[int]] = []
    for prompt_id, _ in numeric_items:
        if prompt_id == "instance_label":
            continue
        boxes.append(full_box)

    return boxes or [full_box]


# --------------------------------------------------------------------------- #
def load_item(npz_file: str, preproc) -> Dict[str, Any]:
    """
    Read a single .npz, build cube boxes (binary masks), apply zoom transform,
    and return the dict expected by `infer_case`.
    """
    npz: Dict[str, Any] = np.load(npz_file, allow_pickle=True)
    imgs = npz["imgs"]  # (1,H,W,D)
    imgs = preproc.preprocess_ct_case(imgs)

    # ────────────────── bounding-box logic ─────────────────── #
    if "boxes" in npz:
        boxes_src = list(npz["boxes"])
    else:
        prompt_dict = npz.get("prompt", None)
        boxes_src = boxes_from_prompt(imgs, prompt_dict)

    # convert every (list | dict) → dict → binary cube
    cube_masks = []
    for b in boxes_src:
        if isinstance(b, list):  # [zmin,ymin,xmin,zmax,ymax,xmax]
            b = {
                "z_min":        b[0],
                "z_mid_y_min":  b[1],
                "z_mid_x_min":  b[2],
                "z_max":        b[3],
                "z_mid_y_max":  b[4],
                "z_mid_x_max":  b[5],
            }
        cube_masks.append(build_binary_cube_dict(b, imgs.shape[1:]))

    cube_boxes = torch.stack(cube_masks, 0)
    assert cube_boxes.shape[1:] == imgs.shape[1:], (
        f"{cube_boxes.shape=} vs {imgs.shape=}"
    )

    # build final dict expected by model’s processor
    item = preproc.zoom_transform_case(imgs, cube_boxes)
    item["file_path"]     = npz_file
    item["img_original"]  = torch.from_numpy(npz["imgs"])
    return item


def backfill_foreground_preds(
    ct_shape: tuple[int, int, int],
    logits_mask: torch.Tensor,
    start_coord: List[int],
    end_coord: List[int],
) -> torch.Tensor:
    """Insert zoomed logits back into full-resolution blank volume."""
    binary = torch.zeros(ct_shape)
    binary[
        start_coord[0] : end_coord[0],
        start_coord[1] : end_coord[1],
        start_coord[2] : end_coord[2],
    ] = torch.sigmoid(logits_mask)
    return (binary > 0.5).float()


@torch.no_grad()
def infer_case(
    model: SegVolModel,
    data_item: Dict[str, Any],
    processor,
    device: torch.device,
) -> np.ndarray:
    """Run model on a single CT volume, return final seg array."""
    data_item["image"]          = data_item["image"].unsqueeze(0).to(device)
    data_item["zoom_out_image"] = data_item["zoom_out_image"].unsqueeze(0).to(device)

    start_c, end_c = data_item["foreground_start_coord"], data_item["foreground_end_coord"]
    img_original   = data_item["img_original"]

    n_classes = data_item["cube_boxes"].shape[0]
    final_preds = torch.zeros_like(img_original)

    for cls_idx in range(n_classes):
        cube_box   = data_item["cube_boxes"][cls_idx].unsqueeze(0).unsqueeze(0)
        bbox_prompt = processor.bbox_prompt_b(
            data_item["zoom_out_cube_boxes"][cls_idx], device=device
        )
        logits_mask = model.forward_test(
            image=data_item["image"],
            zoomed_image=data_item["zoom_out_image"],
            bbox_prompt_group=[bbox_prompt, cube_box],
            use_zoom=True,
        )
        preds = backfill_foreground_preds(img_original.shape, logits_mask, start_c, end_c)
        final_preds[preds == 1] = cls_idx + 1
        torch.cuda.empty_cache()

    return final_preds.cpu().numpy()


# --------------------------------------------------------------------------- #
# ---------------------------------  MAIN  ---------------------------------- #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_dir",   type=str, default="./segvol")
    p.add_argument("--ckpt",        type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--data_glob",   type=str, required=True,
                   help="Glob pattern for input .npz files")
    p.add_argument("--out_dir",     type=str, default="./outputs")
    p.add_argument("--device",      type=str, default="cuda:0",
                   help="'cpu' or 'cuda[:idx]'")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ─────────── build model ─────────── #
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer  = AutoTokenizer.from_pretrained(args.model_dir)
    cfg        = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model      = SegVolModel(cfg)
    model.model.text_encoder.tokenizer = tokenizer
    state_dict = torch.load(args.ckpt, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval().to(device)
    processor  = model.processor

    # ─────────── iterate CTs ─────────── #
    files = sorted(glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.data_glob}")

    with tqdm(total=len(files), desc="Predict") as bar:
        for f in files:
            item = load_item(f, processor)
            segs = infer_case(model, item, processor, device)
            np.savez_compressed(
                os.path.join(args.out_dir, os.path.basename(f)), segs=segs
            )
            if args.verbose:
                bar.write(f"✓ {os.path.basename(f)}  → classes={int(segs.max())}")
            bar.update(1)

    print(f"Done. {len(files)} volumes processed → {args.out_dir}")


if __name__ == "__main__":
    main()