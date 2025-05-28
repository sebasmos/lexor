#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametrised 3-D inference.

Usage
-----
CUDA_VISIBLE_DEVICES=1 python infer_case_parametrized.py \
    --data_glob "/home/sebastian/codes/data/CVPR-2025-CHALLENGE/3D_val_npz/3D_val_npz/*.npz" \
    --model_dir ./segvol \
    --ckpt ./ckpts_mobilenet_2_5d/epoch_100_loss_0.2663_mobilenet_2_5d.pth \
    --out_dir ./outputs_epoch_100_loss_0.2663_mobilenet_2_5d_v2 \
    --device cuda:0
"""

from __future__ import annotations
import os, argparse
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel


# ───────────────────────────── helpers ───────────────────────────── #
def full_volume_box(ct_shape: tuple[int, int, int]) -> List[int]:
    """Return a list [z_min, y_min, x_min, z_max, y_max, x_max] for whole image."""
    _, H, W, D = ct_shape  # (1,H,W,D)
    return [0, 0, 0, H, W, D]


def build_cube_stack(boxes: List[Any], vol_shape: tuple[int, int, int]) -> torch.Tensor:
    """Convert a list[dict|list] → stacked binary cube tensor (C,H,W,D)."""
    cubes: List[torch.Tensor] = []
    for b in boxes:
        if isinstance(b, list):  # flat 6-tuple
            b = {
                "z_min":        b[0],
                "z_mid_y_min":  b[1],
                "z_mid_x_min":  b[2],
                "z_max":        b[3],
                "z_mid_y_max":  b[4],
                "z_mid_x_max":  b[5],
            }
        cubes.append(build_binary_cube_dict(b, vol_shape))
    return torch.stack(cubes, 0)


def load_item(npz_path: str, proc) -> Dict[str, Any]:
    """Load .npz → dict compatible with SegVolModel.processor"""
    npz = np.load(npz_path, allow_pickle=True)
    imgs = npz["imgs"]                # (1,H,W,D)
    imgs = proc.preprocess_ct_case(imgs)

    # ───── bounding boxes ───── #
    if "boxes" in npz and len(npz["boxes"]):
        boxes_src: List[Any] = list(npz["boxes"])
    else:
        # ⇢ best generic fallback: one prompt covering the entire volume
        boxes_src = [full_volume_box(imgs.shape)]

    cube_boxes = build_cube_stack(boxes_src, imgs.shape[1:])
    assert cube_boxes.shape[1:] == imgs.shape[1:], (
        f"{cube_boxes.shape=} vs {imgs.shape=}"
    )

    item = proc.zoom_transform_case(imgs, cube_boxes)
    item["file_path"]     = npz_path
    item["img_original"]  = torch.from_numpy(npz["imgs"])
    return item


def backfill_foreground(
    ct_shape: tuple[int, int, int],
    logits_mask: torch.Tensor,
    start: List[int],
    end: List[int],
) -> torch.Tensor:
    out = torch.zeros(ct_shape)
    out[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = torch.sigmoid(logits_mask)
    return (out > 0.5).float()


@torch.no_grad()
def infer_volume(
    model: SegVolModel,
    sample: Dict[str, Any],
    proc,
    device: torch.device,
) -> np.ndarray:
    sample["image"]          = sample["image"].unsqueeze(0).to(device)
    sample["zoom_out_image"] = sample["zoom_out_image"].unsqueeze(0).to(device)

    s_coord, e_coord = sample["foreground_start_coord"], sample["foreground_end_coord"]
    img_original     = sample["img_original"]

    n_cls   = sample["cube_boxes"].shape[0]
    seg_vol = torch.zeros_like(img_original)

    for cls_idx in range(n_cls):
        cube = sample["cube_boxes"][cls_idx].unsqueeze(0).unsqueeze(0)
        prompt = proc.bbox_prompt_b(sample["zoom_out_cube_boxes"][cls_idx], device=device)

        logits = model.forward_test(
            image=sample["image"],
            zoomed_image=sample["zoom_out_image"],
            bbox_prompt_group=[prompt, cube],
            use_zoom=True,
        )
        region = backfill_foreground(img_original.shape, logits, s_coord, e_coord)
        seg_vol[region == 1] = cls_idx + 1
        torch.cuda.empty_cache()

    return seg_vol.cpu().numpy()


# ─────────────────────────────── main ────────────────────────────── #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_glob", required=True, type=str,
                   help="Glob pattern for input .npz files.")
    p.add_argument("--model_dir", required=True, type=str)
    p.add_argument("--ckpt",      required=True, type=str,
                   help="Path to .pth checkpoint.")
    p.add_argument("--out_dir",   default="./outputs", type=str)
    p.add_argument("--device",    default="cuda:0", type=str)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ───── model & processor ───── #
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    cfg       = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model     = SegVolModel(cfg)
    model.model.text_encoder.tokenizer = tokenizer
    state     = torch.load(args.ckpt, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)
    proc      = model.processor

    # ───── iterate volumes ───── #
    files = sorted(glob(args.data_glob))
    if not files:
        raise RuntimeError(f"No files matched {args.data_glob}")

    with tqdm(total=len(files), desc="Inference") as bar:
        for f in files:
            sample  = load_item(f, proc)
            segs_np = infer_volume(model, sample, proc, device)
            np.savez_compressed(Path(args.out_dir) / Path(f).name, segs=segs_np)
            bar.update(1)

    print(f"Finished. {len(files)} volumes → '{args.out_dir}'")


if __name__ == "__main__":
    main()