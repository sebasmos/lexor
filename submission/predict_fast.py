#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------
# MUST set this *before* importing torch so that allocator obeys it
import os, gc, json, numpy as np
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

from transformers import AutoTokenizer, AutoConfig
import torch
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel
from tqdm import tqdm
import argparse

# ------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True        # safe precision cut
device = torch.device("cuda:0")
output_dir = "output_latest"
os.makedirs(output_dir, exist_ok=True)

# --------------------------  DATASET  -----------------------------
class SegDatasetTest(torch.utils.data.Dataset):
    def __init__(self, file_paths, preprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        npz       = np.load(file_path, allow_pickle=True)
        imgs, gts = npz["imgs"], npz["gts"]

        imgs, gts = self.preprocessor.preprocess_ct_gt(imgs, gts)
        cube_list = [
            build_binary_cube_dict(box, imgs.shape[1:]) for box in npz["boxes"]
        ]
        cube_boxes = torch.stack(cube_list, dim=0)
        assert cube_boxes.shape == gts.shape

        item = self.preprocessor.zoom_transform(imgs, gts, cube_boxes)
        item["file_path"] = file_path
        return item

# --------------------------  VALIDATE  ----------------------------
@torch.no_grad()
def validate(model, loader, proc, max_samples=None):
    tot_dice, tot_vols = 0.0, 0
    for idx, (data_item,) in enumerate(tqdm(loader, desc="Validating")):
        if max_samples and idx >= max_samples: break

        # Pull needed tensors to GPU just for this sample
        image         = data_item["image"        ].unsqueeze(0).to(device)
        zoom_image    = data_item["zoom_out_image"].unsqueeze(0).to(device)
        label_stack   = data_item["label"        ].unsqueeze(0).to(device)
        cube_stack    = data_item["cube_boxes"]                # CPU
        zoom_cubes    = data_item["zoom_out_cube_boxes"]       # CPU
        n_classes     = label_stack.size(1)

        dice_vals = []
        for cls in range(n_classes):
            gt_label   = label_stack[0, cls].to(device, dtype=torch.bool)
            cube_boxes = cube_stack[cls].unsqueeze(0).unsqueeze(0)
            bbox_pr    = proc.bbox_prompt_b(zoom_cubes[cls], device=device)

            logits     = model.forward_test(
                image=image,
                zoomed_image=zoom_image,
                bbox_prompt_group=[bbox_pr, cube_boxes],
                use_zoom=True,
            )
            pred       = logits[0][0]
            dice       = proc.dice_score(pred, gt_label, device)
            dice_vals.append(dice.cpu().item())

            # per-class cleanup
            del gt_label, pred, logits, bbox_pr, cube_boxes
            torch.cuda.empty_cache()

        # ---- store volume-level result (CPU) ----------------------
        mean_dice = float(np.mean(dice_vals))
        print(f"[{idx}] Dice: {mean_dice:.4f}")

        base   = os.path.splitext(os.path.basename(data_item["file_path"]))[0]
        out_np = np.asarray(dice_vals, dtype=np.float32)
        np.savez_compressed(os.path.join(output_dir, f"{base}.npz"), dice=out_np)

        # stats
        tot_dice += mean_dice
        tot_vols += 1

        # ----------- BIG scrub: drop *everything* -----------------
        del image, zoom_image, label_stack, cube_stack, zoom_cubes, data_item
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()   # cleans inter-process cached blocks

    print(f"Average Dice over {tot_vols} vols: {tot_dice / max(tot_vols,1):.4f}")

# ---------------------------  MAIN  ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=None,
                    help="debug: limit #volumes processed")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_dir = "./segvol"
    ckpt      = "./epoch_2000_loss_0.2232.pth"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cfg       = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model = SegVolModel(cfg)
    model.model.text_encoder.tokenizer = tokenizer
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    model.eval().to(device)

    with open("val_samples.json") as f:
        val_paths = json.load(f)
    ds      = SegDatasetTest(val_paths, model.processor)
    loader  = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=lambda x: x
    )

    validate(model, loader, model.processor, args.max_samples)

if __name__ == "__main__":
    main()