"""
SegVol DDP training script
--------------------------------
• Adds memory-friendly tweaks (items 1-4 from the earlier list).  
• **No** mixed-precision.  
• **Gradient-checkpointing (item 5) intentionally omitted.**
"""

from transformers import AutoTokenizer, AutoConfig
import torch
import os
import glob
import numpy as np
from segvol.model_segvol_single import build_overlap_mask, build_binary_cube_dict, SegVolModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
from validation import validation, SegDatasetTest

# ────────────────────────────────────────────────────────────────────────────────
#  utils
# ────────────────────────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path, device, model, optimizer):
    """
    Item 4:  avoid double-buffering – delete the checkpoint once weights are copied.
    """
    if not os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    print(f'load checkpoint from {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device)

    # standard case
    if 'model_state_dict' in ckpt:
        model.module.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', -1) + 1
        best_loss   = ckpt.get('loss', float('inf'))
        if local_rank == 0:
            print(f"Resuming training from epoch {start_epoch}")
    else:  # SegVol_v1.pth format
        model_dict = {k.replace('module.', 'model.'): v
                      for k, v in ckpt['model'].items()}
        model.module.load_state_dict(model_dict)
        start_epoch, best_loss = 0, float('inf')
        if local_rank == 0:
            print("Resuming training from SegVol_v1.pth")

    # free host RAM / cached GPU blocks
    del ckpt
    torch.cuda.empty_cache()
    return start_epoch, best_loss


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ────────────────────────────────────────────────────────────────────────────────
#  dataset
# ────────────────────────────────────────────────────────────────────────────────
class SegDatasetTrain(torch.utils.data.Dataset):
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
        imgs, gts = npz['imgs'], npz['gts']

        imgs = self.preprocessor.preprocess_ct(imgs)
        unique_labs = np.unique(gts)[1:]
        selected_lab = unique_labs[self.epoch % len(unique_labs)]
        gts = self.preprocessor.preprocess_gt(gts, selected_lab)

        return self.preprocessor.train_transform(imgs, gts)


def collate_fn(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'foreground_start_coord': [b['foreground_start_coord'] for b in batch],
        'foreground_end_coord':   [b['foreground_end_coord']   for b in batch],
    }


# ────────────────────────────────────────────────────────────────────────────────
#  validation
# ────────────────────────────────────────────────────────────────────────────────
def val_model(local_rank, model_dir, ckpt_path):
    """
    Item 2: use torch.inference_mode() so no graph is built, freeing VRAM fast.
    """
    if local_rank != 0:
        return 0.0

    print(f'validation on {ckpt_path}')
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, test_mode=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor

    device = torch.device(f"cuda:{local_rank}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model_val.load_state_dict(ckpt['model_state_dict'])
    else:
        model_val.load_state_dict(
            {k.replace('module.', 'model.'): v for k, v in ckpt['model'].items()}
        )
    del ckpt
    model_val.to(device)
    model_val.eval()

    with open('val_samples.json', 'r') as f:
        val_file_paths = json.load(f)
    val_dataset    = SegDatasetTest(val_file_paths, processor)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x
    )

    with torch.inference_mode():
        avg_bbox_dice = validation(model_val, val_dataloader, processor)
    return avg_bbox_dice


# ────────────────────────────────────────────────────────────────────────────────
#  training loop
# ────────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, train_dataloader, optimizer, epoch,
                    empty_cache_every=50):
    """
    Implements all memory-savers (item 1):
      • zero_grad(set_to_none=True)
      • non_blocking transfers
      • del + empty_cache/ipc_collect
    """
    model.train()
    total_loss = 0

    with tqdm(train_dataloader,
              desc=f'Epoch {epoch}',
              disable=(local_rank != 0)) as pbar:
        for step, data_item in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            image = data_item['image'].to(device, non_blocking=True)
            label = data_item['label'].to(device, non_blocking=True)

            loss = model(image=image,
                         train_labels=label[:, 0],
                         train_organs=None)
            loss.backward()
            optimizer.step()

            # stats / progress
            loss_val = loss.item()
            total_loss += loss_val
            if local_rank == 0:
                writer.add_scalar('train/step_Loss',
                                  loss_val,
                                  epoch * len(train_dataloader) + step)
            pbar.set_postfix(loss=f'{loss_val:.4f}',
                             avg_loss=f'{total_loss/(step+1):.4f}')

            # free RAM
            del image, label, loss
            if step % empty_cache_every == 0:
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return total_loss / len(train_dataloader)


# ────────────────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ---------- base params ----------
    batch_size   = 4
    num_workers  = 4
    num_epochs   = 3000
    initial_lr   = 1e-5
    train_root   = '/scratch/vaher/SegFM3D/3D_train_npz_random_10percent_16G'
    resume_ckpt  = 'SegVol_v1.pth'
    save_dir     = './ckpts_segvol_v1_ptrained'
    model_dir    = './segvol'

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(save_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(train_root, '**', '*.npz'),
                            recursive=True)

    # ---------- build model ----------
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir,
                                        trust_remote_code=True,
                                        test_mode=False)
    model = SegVolModel(config)
    model.model.text_encoder.tokenizer = clip_tokenizer
    processor = model.processor     # keep before wrapping

    # ---------- DDP ----------
    global local_rank            # used inside helper fns
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True)
    if local_rank == 0:
        print('model load done')

    # ---------- optimiser ----------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=initial_lr,
                                  weight_decay=1e-5)

    # ---------- resume ----------
    start_epoch, best_loss = load_checkpoint(resume_ckpt,
                                             device,
                                             model,
                                             optimizer)

    # ---------- dataset ----------
    train_dataset = SegDatasetTrain(train_files, processor)
    train_sampler = DistributedSampler(train_dataset)

    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        # item 3: keep workers alive & lower prefetch
        persistent_workers=True,
        prefetch_factor=2
    )
    if local_rank == 0:
        print('train dataset size:', len(train_dataset))
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    else:
        writer = None

    # ---------- train loop ----------
    print(f'start training from epoch {start_epoch}')
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr:.10f}')

        avg_loss = train_one_epoch(model, train_loader,
                                   optimizer, epoch)

        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch} - avg loss {avg_loss:.4f}')

            writer.add_scalar('train/Learning_Rate', current_lr, epoch)
            writer.add_scalar('train/avg_loss',      avg_loss,   epoch)

            # save checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(ckpt, os.path.join(save_dir, 'latest.pth'))

            # best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, 'best_model.pth'))

            # periodic  + validation
            if (epoch + 1) % 25 == 0:
                pth = os.path.join(save_dir,
                                   f'epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
                torch.save(ckpt, pth)
                print(f'save checkpoint to {pth}')
                dice = val_model(local_rank, model_dir, pth)
                writer.add_scalar('validation/BBox_Dice', dice, epoch)
                print(f'Epoch {epoch} - val BBox Dice: {dice:.4f}')

    # ---------- cleanup ----------
    if local_rank == 0:
        writer.close()
    dist.destroy_process_group()
