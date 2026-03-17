# train.py
import argparse
import os
import re
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# 优化 CUDA 算子执行：强制使用 math SDP 以确保数值稳定性（特别是较旧 GPU 或特定 CUDA 版本）
# 注意：这会降低性能和增加显存使用，但兼容性更好
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL

from sit import SiT_models
from loss import SILoss
from dataset import HanziFontDiffusionDatasetWithIndexAndMoments
from sampling_utils import generate_hanzi_grid_flow_matching
from PIL import Image

# -------------------------------------------------
# 辅助工具函数
# -------------------------------------------------
def save_checkpoint(save_dir, epoch, accelerator, model, optimizer, ema=None, scheduler=None):
    """
    保存训练检查点。
    关键修正：使用 accelerator.unwrap_model 获取原始模型状态，避免保存 DDP 包装的前缀。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取去包装后的原始模型状态字典
    unwrapped_model = accelerator.unwrap_model(model)
    
    ckpt = {
        "epoch": epoch,
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
    
    # 只在主进程保存
    if accelerator.is_main_process:
        torch.save(ckpt, ckpt_path)
        print(f"[Checkpoint] Saved checkpoint at epoch {epoch} -> {ckpt_path}")

def load_latest_checkpoint(save_dir, model, optimizer, ema=None, scheduler=None, device="cuda"):
    """
    加载最新的检查点。
    支持加载包含 'module.' 前缀（DDP 保存）或不包含（单卡保存）的状态字典。
    """
    if not os.path.exists(save_dir):
        print("[Checkpoint] No directory found. Starting from scratch.")
        return 0, 0
    
    ckpts = [f for f in os.listdir(save_dir) if re.match(r"epoch_\d+\.pt", f)]
    if len(ckpts) == 0:
        return 0, 0
    
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    latest_ckpt = ckpts[-1]
    ckpt_path = os.path.join(save_dir, latest_ckpt)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 加载模型状态，自动处理 'module.' 前缀
    state_dict = checkpoint["model"]
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的键（如 DDP 前缀）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 如果当前模型没有 'module.' 但检查点有，移除前缀
        name = k.replace("module.", "") if k.startswith("module.") else k
        if name in model_dict:
            new_state_dict[name] = v
        else:
            print(f"[Warning] Key {k} (mapped to {name}) not found in model")
    
    model.load_state_dict(new_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    
    # 加载 scheduler 状态
    scheduler_last_epoch = 0
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        scheduler_last_epoch = checkpoint.get("epoch", 0) - 1
    
    loaded_epoch = checkpoint.get("epoch", 0)
    print(f"[Checkpoint] Loaded checkpoint from epoch {loaded_epoch}")
    return loaded_epoch, scheduler_last_epoch

@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    """
    更新 EMA 模型参数。
    关键修正：确保处理被 DDP 包装的模型（通过传入 unwrap_model 后的模型）。
    """
    ema_params = OrderedDict(ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # 清理名称中的 'module.' 前缀（如果存在）
        clean_name = name.replace("module.", "")
        if clean_name in ema_params:
            ema_params[clean_name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            # 如果 EMA 中没有对应参数，跳过（可能是新添加的参数）
            pass

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# 数据增强/转换
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# -------------------------------------------------
# 主训练流程
# -------------------------------------------------
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if args.seed is not None:
        # 不同进程使用不同种子以避免数据加载同步
        set_seed(args.seed + accelerator.process_index)

    # 1. Dataset
    dataset = HanziFontDiffusionDatasetWithIndexAndMoments(
        root_dir=args.data_dir,
        feature_path=args.feature_path,
        moments_path=args.moments_path,
        transform=transform_224,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # 2. Model 初始化
    latent_size = args.resolution // 8  # 对于 224px 输入，latent 为 28
    model = SiT_models[args.model](
        input_size=latent_size
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # VAE 用于 Latent 空间转换
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=False
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    # 3. Loss 初始化
    loss_fn = SILoss(
        path_type="linear",
        time_sampler="logit_normal",
        label_dropout_prob=args.cfg_prob,  # CFG dropout 概率
        weighting="uniform",
    )

    # 4. Accelerate 准备
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 学习率调度器 (需要在 prepare 之后创建，因为需要知道 dataloader 长度)
    scheduler = None
    if args.scheduler != "none":
        total_steps = args.epochs * len(dataloader)
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        if args.scheduler == "cosine":
            # Cosine Annealing with Warmup
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            print(f"[Scheduler] Using Cosine Annealing with {warmup_steps} warmup steps (total: {total_steps})")
        elif args.scheduler == "constant":
            # Constant learning rate with warmup
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            print(f"[Scheduler] Using Constant LR with {warmup_steps} warmup steps")
    
    # 加载权重
    start_epoch, _ = load_latest_checkpoint(args.out_dir, model, optimizer, ema=ema, scheduler=scheduler, device=device)
    
    if start_epoch == 0:
        # 初始 EMA 与模型同步
        raw_model = accelerator.unwrap_model(model)
        update_ema(ema, raw_model, decay=0.0)

    # SD VAE 标准缩放系数
    # 注意：此系数应用于 VAE 编码后的 latent，解码时需除以相同系数
    latents_scale = 0.18215
    
    global_step = 0
    model.train()

    # 5. 训练循环
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_total_loss = 0.0
        epoch_step_count = 0
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

        for batch in pbar:
            z_glyph = batch["z_glyph"].to(device)
            z_style = batch["z_style"].to(device)
            moments = batch["moments"].to(device)

            # 从 Moments 重采样出 Latent
            # moments 应包含 [mean, log_var] 或预计算的 VAE 编码统计量
            with torch.no_grad():
                posterior = DiagonalGaussianDistribution(moments)
                latents = posterior.sample() * latents_scale

            with accelerator.accumulate(model):
                model_kwargs = dict(z_glyph=z_glyph, z_style=z_style)
                
                # 计算 Flow Matching Loss
                loss, loss_ref = loss_fn(model, latents, model_kwargs)
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # 更新学习率调度器
                if scheduler is not None:
                    scheduler.step()

                if accelerator.sync_gradients:
                    # 关键修正：使用 unwrap_model 获取原始模型更新 EMA
                    raw_model = accelerator.unwrap_model(model)
                    update_ema(ema, raw_model)
                    
                    epoch_total_loss += loss_ref.detach().item()
                    epoch_step_count += 1
                    global_step += 1
                    pbar.set_postfix(mse=f"{loss_ref.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        # Epoch 结束：记录与采样
        if accelerator.is_main_process:
            avg_loss = epoch_total_loss / max(1, epoch_step_count)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch + 1}] Avg Loss: {avg_loss:.6f}, LR: {current_lr:.6e}")

            # 采样可视化
            sample_dir = os.path.join(args.out_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 使用 EMA 模型进行采样，并设置为评估模式
            sample_model = ema
            sample_model.eval()
            
            # 动态获取有效的字体 ID（避免硬编码超出数据集范围）
            dataset_size = len(dataset)
            font_ids = [min(i, dataset_size - 1) for i in [0, 100, 500, 1000, 2000, 3000, 4000, 5000] if i < dataset_size]
            if len(font_ids) < 4:
                # 如果数据集太小，使用前 N 个样本
                font_ids = list(range(min(8, dataset_size)))
            
            style_id = min(12, dataset_size - 1) if dataset_size > 12 else 0

            try:
                with torch.no_grad():
                    _, grid_image = generate_hanzi_grid_flow_matching(
                        model=sample_model, 
                        vae=vae, 
                        dataset=dataset,
                        device=device, 
                        font_ids=font_ids, 
                        style_id=style_id, 
                        cfg_scale=7.5,
                    )
                Image.fromarray(grid_image).save(os.path.join(sample_dir, f"epoch_{epoch+1:04d}.png"))
                print(f"[Sample] Saved visualization for epoch {epoch + 1}.")
            except Exception as e:
                print(f"[Warning] Sampling failed: {e}")
                import traceback
                traceback.print_exc()

            # 保存 Checkpoint（传入 accelerator 用于 unwrap_model）
            save_checkpoint(args.out_dir, epoch + 1, accelerator, model, optimizer, ema=ema, scheduler=scheduler)

    accelerator.wait_for_everyone()
    accelerator.end_training()

def parse_args():
    parser = argparse.ArgumentParser(description="Train SiT for Chinese Font Generation")
    parser.add_argument("--data-dir", type=str, default="fonts/train", help="训练数据目录")
    parser.add_argument("--feature-path", type=str, default="features/hanzi_font_clip_features.pt", help="字形特征路径")
    parser.add_argument("--moments-path", type=str, default="fonts/moments.pt", help="VAE moments 缓存路径")
    parser.add_argument("--resolution", type=int, default=224, help="输入图像分辨率")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="学习率最小值 (cosine scheduler)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "constant"], help="学习率调度器类型")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup 步数占总步数的比例")
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载 workers")
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="CFG dropout 概率")
    parser.add_argument("--model", type=str, default="SiT-B/2", choices=list(SiT_models.keys()), help="模型架构")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="混合精度模式")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="输出目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)