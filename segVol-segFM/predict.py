from transformers import AutoTokenizer, AutoConfig
import torch
import os
import numpy as np
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel
from tqdm import tqdm
import json

device_id = 0
device = torch.device(f"cuda:{device_id}")

# ──────────────────────────────────────────────────────────────────
# NEW ➜  make sure output_latest/ exists
output_dir = 'output_latest'
os.makedirs(output_dir, exist_ok=True)
# ──────────────────────────────────────────────────────────────────

class SegDatasetTest(torch.utils.data.Dataset):
    def __init__(self, file_paths, preprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        self.epoch = 0
    
    def __len__(self):
        return len(self.file_paths)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        npz = np.load(file_path, allow_pickle=True)
        imgs = npz['imgs']
        gts = npz['gts']
        
        imgs, gts = self.preprocessor.preprocess_ct_gt(imgs, gts)   # (1, H, W, D), (N, H, W, D)
        boxes = npz['boxes']  # list of bounding-box prompts
        cube_boxes = []
        for std_box in boxes:
            binary_cube = build_binary_cube_dict(std_box, imgs.shape[1:])
            cube_boxes.append(binary_cube)
        cube_boxes = torch.stack(cube_boxes, dim=0)
        assert cube_boxes.shape == gts.shape, f'{cube_boxes.shape} != {gts.shape}'

        zoom_item = self.preprocessor.zoom_transform(imgs, gts, cube_boxes)
        zoom_item['file_path'] = file_path    # keep original path for naming the output
        return zoom_item

def validation(model_val, val_dataloader, processor):

    total_bbox_dice = 0.0
    total_samples = 0
    idx = 0
    for data_item_list in tqdm(val_dataloader, desc="Validating"):
        data_item = data_item_list[0]
        category_num = data_item['label'].shape[0]
        batch_bbox_dice = 0.0
        print(data_item['file_path'])
        data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
            data_item['image'].unsqueeze(0).to(device), \
            data_item['label'].unsqueeze(0).to(device), \
            data_item['zoom_out_image'].unsqueeze(0).to(device), \
            data_item['zoom_out_label'].unsqueeze(0).to(device)
        
        # ---- collect predicted masks for *all* classes so we can save once per volume
        pred_masks = torch.zeros_like(data_item['label'][0], dtype=torch.bool)
        bbox_dice_list = []
        for cls_idx in range(category_num):
            gt_label = data_item['label'][0][cls_idx].to(device, dtype=torch.bool, non_blocking=True)

            cube_boxes = data_item['cube_boxes'][cls_idx].unsqueeze(0).unsqueeze(0)
            bbox_prompt = processor.bbox_prompt_b(
                data_item['zoom_out_cube_boxes'][cls_idx], device=device
            )

            with torch.no_grad():
                logits_mask = model_val.forward_test(
                    image               = data_item['image'],          # still float-32/16 on GPU
                    zoomed_image        = data_item['zoom_out_image'],
                    bbox_prompt_group   = [bbox_prompt, cube_boxes],
                    use_zoom            = True
                )

            pred_label = logits_mask[0][0]             # bool
            bbox_dice = processor.dice_score(pred_label, gt_label, device)
            
            bbox_dice_list.append(bbox_dice)
            
            
            
            del gt_label, pred_label, logits_mask
            torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()
        mean_bbox_dice = torch.stack(bbox_dice_list).mean()

        print(f'[{idx}/{len(val_dataloader)}]bbox_dice: {mean_bbox_dice:.4f}')
        # ---- NEW ➜ save every volume once, keeping the same base-name
        base_name = os.path.splitext(os.path.basename(data_item['file_path']))[0]
        out_path  = os.path.join(output_dir, f"{base_name}.npz")
        np.savez_compressed(out_path, preds=pred_masks.cpu().numpy())
        # ----

        total_bbox_dice += batch_bbox_dice
        total_samples   += category_num
        idx += 1

    avg_bbox_dice = total_bbox_dice / total_samples
    print(f'Average BBox Dice: {avg_bbox_dice:.4f}')
    return avg_bbox_dice

if __name__ == '__main__':
    model_dir = './segvol'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ckpt_path = './epoch_2000_loss_0.2232.pth'
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_val.load_state_dict(checkpoint['model_state_dict'])
    model_val.eval()
    model_val.to(device)

    with open('val_samples.json', 'r') as f:
        val_file_paths = json.load(f)
    
    val_dataset = SegDatasetTest(val_file_paths, processor)
    print('val dataset size:', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x
    )
    validation(model_val, val_dataloader, processor)