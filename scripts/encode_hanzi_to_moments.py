import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL

# 设置 HF endpoint 为镜像（可选）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def get_transform():
    """
    获取图像预处理 transform（必须匹配 SD-VAE 期望）
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def encode_images_to_moments(
    vae,
    dataloader,
    device="cuda",
    description="Encoding images with VAE"
):
    """
    将数据加载器中的图像编码为 VAE latent moments

    参数:
        vae: 预训练的 VAE 模型
        dataloader: 数据加载器
        device: 计算设备
        description: 进度条描述

    返回:
        dict: 包含 'mean' 和 'logvar' 的字典
    """
    all_means = []
    all_logvars = []

    pbar = tqdm(dataloader, desc=description)

    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(imgs).latent_dist
            mean = posterior.mean          # [B, C, H, W]
            logvar = posterior.logvar      # [B, C, H, W]

        all_means.append(mean.cpu())
        all_logvars.append(logvar.cpu())

    # 拼接并保存
    moments = {
        "mean": torch.cat(all_means, dim=0),
        "logvar": torch.cat(all_logvars, dim=0),
    }

    return moments


def main(args):
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # 确保必要的文件存在
    if not os.path.exists(args.vae_path):
        print(f"[ERROR] VAE 模型不存在: {args.vae_path}")
        print("请先下载 VAE 模型或确保路径正确")
        return

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] 数据目录不存在: {args.data_dir}")
        return

    if not os.path.exists(args.feature_path):
        print(f"[ERROR] 特征文件不存在: {args.feature_path}")
        return

    # -------------------------------------------------
    # 1. 加载预训练 VAE
    # -------------------------------------------------
    print("[VAE] Loading Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        local_files_only=True
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print("[VAE] Loaded successfully")

    # -------------------------------------------------
    # 2. 加载数据集
    # -------------------------------------------------
    print("[Dataset] Loading dataset...")
    transform = get_transform()

    # 根据数据集类型选择合适的类
    if args.dataset_type == "with_moments":
        from dataset import HanziFontDiffusionDatasetWithIndexAndMoments as DatasetClass
    else:
        from dataset import HanziFontDiffusionDatasetWithIndex as DatasetClass

    dataset = DatasetClass(
        root_dir=args.data_dir,
        feature_path=args.feature_path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"[Dataset] Total samples: {len(dataset)}")
    print(f"[Dataset] Batches: {len(dataloader)}")

    # -------------------------------------------------
    # 3. 编码并保存 moments
    # -------------------------------------------------
    print("[Encoding] Starting VAE encoding...")

    moments = encode_images_to_moments(
        vae=vae,
        dataloader=dataloader,
        device=device,
        description="Encoding images with VAE"
    )

    # 保存
    torch.save(moments, args.output_path)

    print(f"\n[Success] Saved moments to {args.output_path}")
    print(f"  mean shape:   {moments['mean'].shape}")
    print(f"  logvar shape: {moments['logvar'].shape}")

    # -------------------------------------------------
    # 4. 验证（可选）
    # -------------------------------------------------
    if args.verify:
        print("\n[Verify] Testing loaded moments...")
        try:
            from dataset import HanziFontDiffusionDatasetWithIndexAndMoments

            verify_dataset = HanziFontDiffusionDatasetWithIndexAndMoments(
                root_dir=args.data_dir,
                feature_path=args.feature_path,
                moments_path=args.output_path,
                transform=transform
            )

            sample = verify_dataset[0]
            print(f"  Verified moments shape: {sample['moments'].shape}")
            print("[Verify] Success!")
        except Exception as e:
            print(f"[Verify] Warning: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将中文字体图像编码为 VAE latent moments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/encode_hanzi_to_moments.py --data-dir data/fonts/train --output-path moments.pt

  # 指定特征文件和批次大小
  python scripts/encode_hanzi_to_moments.py \\
      --data-dir data/fonts/train \\
      --feature-path features/hanzi_font_clip_features.pt \\
      --output-path moments.pt \\
      --batch-size 128

  # 验证编码结果
  python scripts/encode_hanzi_to_moments.py \\
      --data-dir data/fonts/train \\
      --output-path moments.pt \\
      --verify
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fonts/train",
        help="字体图像数据目录 (默认: data/fonts/train)"
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        default="features/hanzi_font_clip_features.pt",
        help="CLIP 特征文件路径 (默认: features/hanzi_font_clip_features.pt)"
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default="./models/stabilityai/sd-vae-ft-mse",
        help="预训练 VAE 模型路径 (默认: ./models/stabilityai/sd-vae-ft-mse)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="moments.pt",
        help="输出 moments 文件路径 (默认: moments.pt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批次大小 (默认: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载线程数 (默认: 4)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="index",
        choices=["index", "with_moments"],
        help="数据集类型 (默认: index)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="编码完成后验证结果"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
