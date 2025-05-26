"""
SegVol DDP training script (with debug output for SLURM)
--------------------------------------------------------
• Adds memory-friendly tweaks
• DDP with debug prints for visibility in SLURM
• No mixed-precision
• No gradient checkpointing
"""
import argparse
import time
from segvol.fast_encoders import FastEncoderFactory, EncoderBenchmark
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
import socket
import sys
import traceback





def load_checkpoint(checkpoint_path, device, model, optimizer):
    if not os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Checkpoint not found: {checkpoint_path}", flush=True)
        return 0, float('inf')

    print(f'Loading checkpoint from {checkpoint_path}', flush=True)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}", flush=True)
        return 0, float('inf')

    if 'model_state_dict' in ckpt:
        model.module.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', -1) + 1
        best_loss   = ckpt.get('loss', float('inf'))
        if local_rank == 0:
            print(f"Resuming training from epoch {start_epoch}", flush=True)
    else:
        model_dict = {k.replace('module.', 'model.'): v for k, v in ckpt['model'].items()}
        model.module.load_state_dict(model_dict)
        start_epoch, best_loss = 0, float('inf')
        if local_rank == 0:
            print("Resuming training from SegVol_v1.pth", flush=True)

    del ckpt
    torch.cuda.empty_cache()
    return start_epoch, best_loss


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


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


def val_model(local_rank, model_dir, ckpt_path):
    if local_rank != 0:
        return 0.0

    print(f'Validation on {ckpt_path}', flush=True)
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
        model_val.load_state_dict({k.replace('module.', 'model.'): v for k, v in ckpt['model'].items()})
    del ckpt
    model_val.to(device)
    model_val.eval()

    with open('val_samples.json', 'r') as f:
        val_file_paths = json.load(f)
    val_dataset = SegDatasetTest(val_file_paths, processor)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    with torch.inference_mode():
        avg_bbox_dice = validation(model_val, val_dataloader, processor)
    return avg_bbox_dice


def train_one_epoch(model, train_dataloader, optimizer, epoch, empty_cache_every=50):
    model.train()
    total_loss = 0

    with tqdm(train_dataloader, desc=f'Epoch {epoch}', disable=(local_rank != 0)) as pbar:
        for step, data_item in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            image = data_item['image'].to(device, non_blocking=True)
            label = data_item['label'].to(device, non_blocking=True)

            loss = model(image=image, train_labels=label[:, 0], train_organs=None)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            if local_rank == 0:
                writer.add_scalar('train/step_Loss', loss_val, epoch * len(train_dataloader) + step)
            pbar.set_postfix(loss=f'{loss_val:.4f}', avg_loss=f'{total_loss/(step+1):.4f}')

            del image, label, loss
            if step % empty_cache_every == 0:
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return total_loss / len(train_dataloader)

def parse_args():
    parser = argparse.ArgumentParser(description='SegVol Training with Fast Encoders')
    
    # Your existing parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--initial_lr', type=float, default=1e-5)
    parser.add_argument('--train_root', type=str, 
                       default='/scratch/vaher/SegFM3D/3D_train_npz_random_10percent_16G')
    parser.add_argument('--resume_ckpt', type=str, default='SegVol_v1.pth')
    parser.add_argument('--save_dir', type=str, default='./ckpts_segvol_v1_ptrained')
    parser.add_argument('--model_dir', type=str, default='./segvol')
    
    # Fast encoder parameters
    parser.add_argument('--fast_encoder_type', type=str, default='original',
                       choices=['original', 'fast_vit', 'ultra_fast_vit', 'mobilenet_2_5d', 
                               'efficientnet_2_5d', 'mobilenet_3d', 'hybrid_cnn_vit', 'fast_resnet3d'])
    parser.add_argument('--benchmark_encoders', action='store_true')
    parser.add_argument('--log_inference_time', action='store_true')
    
    return parser.parse_args()


def run_encoder_benchmark(local_rank, device):
    """Run encoder benchmark before training"""
    if local_rank != 0:
        return
    
    print("\n" + "="*80, flush=True)
    print("ENCODER BENCHMARK COMPARISON", flush=True)
    print("="*80, flush=True)
    
    benchmark = EncoderBenchmark(
        input_shape=(1, 1, 32, 256, 256),  # Single sample for fair comparison
        device=device
    )
    
    encoder_types = ['fast_vit', 'ultra_fast_vit', 'mobilenet_2_5d', 'efficientnet_2_5d', 'hybrid_cnn_vit']
    results = benchmark.compare_encoders(encoder_types)
    
    print("\nBenchmark Results:", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Encoder Type':<20} | {'Time (ms)':<10} | {'FPS':<8} | {'Memory (MB)':<12} | Status")
    print("-" * 80, flush=True)
    
    for encoder_type, result in results.items():
        if 'error' in result:
            print(f"{encoder_type:<20} | ERROR: {result['error']}", flush=True)
        else:
            memory_str = f"{result['memory_mb']:.1f}" if result['memory_mb'] else "N/A"
            print(f"{encoder_type:<20} | {result['avg_time_ms']:<10.2f} | "
                  f"{result['fps']:<8.1f} | {memory_str:<12} | OK", flush=True)
    
    print("="*80 + "\n", flush=True)


def benchmark_encoder_speed(model, device, input_shape=(4, 1, 32, 256, 256), num_runs=50):
    """Benchmark encoder speed during training"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.module.model.image_encoder(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.module.model.image_encoder(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    return avg_time


def train_one_epoch_with_timing(model, train_dataloader, optimizer, epoch, local_rank, writer, 
                                log_inference_time=False, empty_cache_every=50):
    """Modified train_one_epoch with timing support"""
    model.train()
    total_loss = 0
    inference_times = []

    with tqdm(train_dataloader, desc=f'Epoch {epoch}', disable=(local_rank != 0)) as pbar:
        for step, data_item in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            image = data_item['image'].to(f"cuda:{local_rank}", non_blocking=True)
            label = data_item['label'].to(f"cuda:{local_rank}", non_blocking=True)

            # Measure inference time if requested
            if log_inference_time and step % 20 == 0:  # Log every 20 steps
                torch.cuda.synchronize()
                start_time = time.time()
                
                loss = model(image=image, train_labels=label[:, 0], train_organs=None)
                
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms
                inference_times.append(inference_time)
            else:
                loss = model(image=image, train_labels=label[:, 0], train_organs=None)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            
            if local_rank == 0:
                writer.add_scalar('train/step_Loss', loss_val, epoch * len(train_dataloader) + step)
                
                # Log inference time
                if log_inference_time and inference_times:
                    avg_inference_time = np.mean(inference_times[-10:])  # Last 10 measurements
                    writer.add_scalar('train/inference_time_ms', avg_inference_time, 
                                    epoch * len(train_dataloader) + step)
            
            pbar.set_postfix(loss=f'{loss_val:.4f}', avg_loss=f'{total_loss/(step+1):.4f}')

            del image, label, loss
            if step % empty_cache_every == 0:
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return total_loss / len(train_dataloader), inference_times


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Extract arguments
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    initial_lr = args.initial_lr
    train_root = args.train_root
    resume_ckpt = args.resume_ckpt
    save_dir = args.save_dir
    model_dir = args.model_dir
    fast_encoder_type = args.fast_encoder_type

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(save_dir, exist_ok=True)

    global local_rank
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    print(f"[{socket.gethostname()}] Global Rank: {dist.get_rank()}, Local Rank: {local_rank}, Device: {device}", flush=True)
    print(f"[{socket.gethostname()}] Using fast encoder: {fast_encoder_type}", flush=True)

    # Run encoder benchmark if requested
    if args.benchmark_encoders:
        run_encoder_benchmark(local_rank, device)

    if local_rank == 0:
        print(f"Checking dataset path: {train_root}", flush=True)
    if not os.path.exists(train_root):
        print(f"[{socket.gethostname()}] [Rank {dist.get_rank()}] ERROR: Dataset path does not exist: {train_root}", flush=True)
        sys.exit(1)

    train_files = glob.glob(os.path.join(train_root, '**', '*.npz'), recursive=True)

    # Load model with fast encoder configuration
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, test_mode=False)
    
    # KEY CHANGE: Set the fast encoder type
    config.fast_encoder_type = fast_encoder_type
    
    model = SegVolModel(config)
    model.model.text_encoder.tokenizer = clip_tokenizer
    processor = model.processor

    if local_rank == 0:
        print(f"Model created with encoder: {fast_encoder_type}", flush=True)
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.model.image_encoder.parameters())
        print(f"Total parameters: {total_params:,}", flush=True)
        print(f"Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)", flush=True)

    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if local_rank == 0:
        print('model load done', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)

    start_epoch, best_loss = load_checkpoint(resume_ckpt, device, model, optimizer)

    train_dataset = SegDatasetTrain(train_files, processor)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    if local_rank == 0:
        print('train dataset size:', len(train_dataset), flush=True)
        # Create tensorboard writer with encoder info
        log_dir = os.path.join(save_dir, 'logs', f'encoder_{fast_encoder_type}')
        writer = SummaryWriter(log_dir)
        
        # Log configuration to tensorboard
        writer.add_text('config/encoder_type', fast_encoder_type, 0)
        writer.add_text('config/batch_size', str(batch_size), 0)
        writer.add_text('config/initial_lr', str(initial_lr), 0)
    else:
        writer = None

    print(f"[{socket.gethostname()}] [Rank {dist.get_rank()}] Starting training loop from epoch {start_epoch}", flush=True)

    # Benchmark encoder speed before training
    if local_rank == 0 and args.log_inference_time:
        initial_speed = benchmark_encoder_speed(model, device)
        print(f"Initial encoder speed: {initial_speed:.2f} ms per batch", flush=True)
        writer.add_scalar('benchmark/encoder_speed_ms', initial_speed, 0)

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch} - Learning rate: {current_lr:.10f} - Encoder: {fast_encoder_type}', flush=True)

        # Use timing-enabled training function if requested
        if args.log_inference_time:
            avg_loss, inference_times = train_one_epoch_with_timing(
                model, train_loader, optimizer, epoch, local_rank, writer, 
                log_inference_time=True
            )
        else:
            avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
            inference_times = []

        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{socket.gethostname()}] [Rank {dist.get_rank()}] Epoch {epoch} - avg loss {avg_loss:.4f} - encoder: {fast_encoder_type}", flush=True)

            writer.add_scalar('train/Learning_Rate', current_lr, epoch)
            writer.add_scalar('train/avg_loss', avg_loss, epoch)
            
            # Log average inference time for this epoch
            if inference_times:
                avg_inference_time = np.mean(inference_times)
                writer.add_scalar('train/epoch_avg_inference_time_ms', avg_inference_time, epoch)
                print(f"Average inference time this epoch: {avg_inference_time:.2f} ms", flush=True)

            # Save checkpoint with encoder type info
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'encoder_type': fast_encoder_type,  # Save encoder type
                'config': vars(args),  # Save all arguments
            }
            torch.save(ckpt, os.path.join(save_dir, 'latest.pth'))

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, f'best_model_{fast_encoder_type}.pth'))

            if (epoch + 1) % 25 == 0:
                pth = os.path.join(save_dir, f'epoch_{epoch+1}_loss_{avg_loss:.4f}_{fast_encoder_type}.pth')
                torch.save(ckpt, pth)
                print(f'save checkpoint to {pth}', flush=True)
                dice = val_model(local_rank, model_dir, pth)
                writer.add_scalar('validation/BBox_Dice', dice, epoch)
                print(f'Epoch {epoch} - val BBox Dice: {dice:.4f} - encoder: {fast_encoder_type}', flush=True)

    if local_rank == 0:
        writer.close()

    print(f"[{socket.gethostname()}] [Rank {dist.get_rank()}] Training finished with encoder: {fast_encoder_type}. Exiting.", flush=True)
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

