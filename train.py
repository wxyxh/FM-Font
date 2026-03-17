import argparse
import os
import re
import gc
from copy import deepcopy
from collections import OrderedDict

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision("high")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL

from sit import SiT_models
from loss import SILoss
from dataset import HanziFontDiffusionDatasetWithIndexAndMoments
from sampling_utils import generate_hanzi_grid_flow_matching
from PIL import Image


def save_checkpoint(save_dir, epoch, accelerator, model, optimizer, lr_scheduler, ema=None):
    os.makedirs(save_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    
    ckpt = {
        "epoch": epoch,
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(), 
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
        
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
    if accelerator.is_main_process:
        torch.save(ckpt, ckpt_path)

def load_latest_checkpoint(save_dir, model, optimizer, lr_scheduler=None, ema=None, device="cuda"):
    if not os.path.exists(save_dir):
        return 0, None
    
    ckpts = [f for f in os.listdir(save_dir) if re.match(r"epoch_\d+\.pt", f)]
    if not ckpts: return 0, None
    
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    ckpt_path = os.path.join(save_dir, ckpts[-1])
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = checkpoint["model"]
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        if name in model_dict:
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    
    loaded_epoch = checkpoint.get("epoch", 0)
    return loaded_epoch, checkpoint

@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    ema_params = OrderedDict(ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        clean_name = name.replace("module.", "")
        if clean_name in ema_params:
            ema_params[clean_name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    log_path = os.path.join(args.out_dir, "train_log.csv")

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    dataset = HanziFontDiffusionDatasetWithIndexAndMoments(
        root_dir=args.data_dir,
        feature_path=args.feature_path,
        moments_path=args.moments_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    latent_size = args.resolution // 8
    model = SiT_models[args.model](input_size=latent_size).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
    )

    start_epoch, raw_ckpt = load_latest_checkpoint(
        args.out_dir, model, optimizer, lr_scheduler, ema, device
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    if start_epoch > 0 and raw_ckpt is not None:
        if "lr_scheduler" not in raw_ckpt:
            steps_to_skip = start_epoch * len(dataloader) // args.grad_accum
            for _ in range(steps_to_skip):
                lr_scheduler.step()
            del raw_ckpt
            gc.collect()
            torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained("./models/stabilityai/sd-vae-ft-mse", local_files_only=True).to(device)
    vae.eval(); vae.requires_grad_(False)
    loss_fn = SILoss(path_type="linear", time_sampler="logit_normal", label_dropout_prob=args.cfg_prob)

    if start_epoch == 0:
        update_ema(ema, accelerator.unwrap_model(model), decay=0.0)

    latents_scale = 0.18215
    model.train()

    if accelerator.is_main_process and start_epoch == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("epoch,avg_loss\n")

    for epoch in range(start_epoch, args.epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        
        epoch_total_loss = 0.0
        epoch_step_count = 0
        micro_loss = 0.0
        micro_step_count = 0
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

        for batch in pbar:
            z_glyph = batch["z_glyph"]
            z_style = batch["z_style"]
            moments = batch["moments"]

            with torch.no_grad():
                posterior = DiagonalGaussianDistribution(moments)
                latents = posterior.sample() * latents_scale

            with accelerator.accumulate(model):
                model_kwargs = dict(z_glyph=z_glyph, z_style=z_style)
                loss, loss_ref = loss_fn(model, latents, model_kwargs)
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, accelerator.unwrap_model(model))

                micro_loss += loss_ref.detach().item()
                micro_step_count += 1
                
                if accelerator.sync_gradients:
                    epoch_step_loss = micro_loss / micro_step_count
                    epoch_total_loss += epoch_step_loss
                    epoch_step_count += 1
                    micro_loss = 0.0
                    micro_step_count = 0
                    
                    curr_lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(mse=f"{epoch_step_loss:.4f}", lr=f"{curr_lr:.2e}")

        if accelerator.is_main_process:
            avg_loss = epoch_total_loss / max(1, epoch_step_count)
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.6f}")
            with open(log_path, "a") as f:
                f.write(f"{epoch + 1},{avg_loss:.6f}\n")
            
            save_checkpoint(args.out_dir, epoch + 1, accelerator, model, optimizer, lr_scheduler, ema)

            sample_dir = os.path.join(args.out_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            ema.eval()
            try:
                dataset_size = len(dataset)
                font_ids = [min(i, dataset_size - 1) for i in [0, 6052, 6053, 6054, 6055, 6056, 6057, 6058]]
                with torch.no_grad():
                    _, grid_image = generate_hanzi_grid_flow_matching(
                        model=ema, vae=vae, dataset=dataset, device=device, 
                        font_ids=font_ids, style_id=12, cfg_scale=7.5)
                Image.fromarray(grid_image).save(os.path.join(sample_dir, f"epoch_{epoch+1:04d}.png"))
            except Exception as e:
                print(f"[Warning] Sampling failed: {e}")
            ema.train()

    accelerator.end_training()

def parse_args():
    parser = argparse.ArgumentParser(description="SiT Training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--feature-path", type=str, required=True)
    parser.add_argument("--moments-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--model", type=str, default="SiT-L/2")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
