import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def get_transform():
    """Get image transform for SD-VAE."""
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
    """Encode images to VAE latent moments."""
    all_means = []
    all_logvars = []

    pbar = tqdm(dataloader, desc=description)

    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(imgs).latent_dist
            mean = posterior.mean
            logvar = posterior.logvar

        all_means.append(mean.cpu())
        all_logvars.append(logvar.cpu())

    moments = {
        "mean": torch.cat(all_means, dim=0),
        "logvar": torch.cat(all_logvars, dim=0),
    }

    return moments


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    if not os.path.exists(args.vae_path):
        print(f"[ERROR] VAE not found: {args.vae_path}")
        return

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data dir not found: {args.data_dir}")
        return

    if not os.path.exists(args.feature_path):
        print(f"[ERROR] Feature file not found: {args.feature_path}")
        return

    # Load VAE
    print("[VAE] Loading Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        local_files_only=True
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print("[VAE] Loaded successfully")

    # Load dataset
    print("[Dataset] Loading dataset...")
    transform = get_transform()

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

    # Encode
    print("[Encoding] Starting VAE encoding...")

    moments = encode_images_to_moments(
        vae=vae,
        dataloader=dataloader,
        device=device,
        description="Encoding images with VAE"
    )

    torch.save(moments, args.output_path)

    print(f"\n[Success] Saved moments to {args.output_path}")
    print(f"  mean shape:   {moments['mean'].shape}")
    print(f"  logvar shape: {moments['logvar'].shape}")

    # Verify
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
        description="Encode hanzi font images to VAE latent moments"
    )

    parser.add_argument("--data-dir", type=str, default="data/fonts/train")
    parser.add_argument("--feature-path", type=str, default="features/hanzi_font_clip_features.pt")
    parser.add_argument("--vae-path", type=str, default="./models/stabilityai/sd-vae-ft-mse")
    parser.add_argument("--output-path", type=str, default="moments.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-type", type=str, default="index", choices=["index", "with_moments"])
    parser.add_argument("--verify", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
