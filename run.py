import os
import re
import sys
import shutil
import glob
import subprocess

def get_latest_checkpoint_epoch(checkpoint_dir):
    """Get latest checkpoint epoch."""
    if not os.path.exists(checkpoint_dir):
        return 0
    
    ckpts = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
    if not ckpts:
        return 0
    
    epochs = []
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+)\.pt", os.path.basename(ckpt))
        if match:
            epochs.append(int(match.group(1)))
    
    return max(epochs) if epochs else 0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("[INFO] Starting Flow Matching training...")
    print()
    
    # Find accelerate
    accelerate_cmd = shutil.which("accelerate")
    if accelerate_cmd is None:
        print("[ERROR] accelerate not found")
        print("       Please install: pip install accelerate")
        sys.exit(1)
    
    print(f"[OK] accelerate: {accelerate_cmd}")
    
    # Check train script
    train_script = os.path.join("", "train.py")
    if not os.path.exists(train_script):
        print(f"[ERROR] Train script not found: {train_script}")
        sys.exit(1) 
    
    # Detect latest checkpoint
    checkpoint_dir = "checkpoints"
    latest_epoch = get_latest_checkpoint_epoch(checkpoint_dir)
    
    target_epochs = 1000
    
    if latest_epoch > 0:
        target_epochs = latest_epoch + target_epochs
        print(f"[INFO] Found checkpoint: epoch {latest_epoch}")
        print(f"[INFO] Resuming from epoch {latest_epoch} to {target_epochs}")
    else:
        print(f"[INFO] No checkpoint found, training from scratch to {target_epochs}")
    
    print("[INFO] Starting training...")
    print()
    
    cmd = [
        "accelerate", "launch",
        train_script,
        "--data-dir", "data/fonts/train",
        "--feature-path", "features/hanzi_font_clip_features.pt",
        "--moments-path", "./moments.pt",
        "--batch-size", "256",
        "--grad-accum", "4",
        "--epochs", str(target_epochs),
        "--lr", "5e-5",
        "--lr-min", "1e-7",
        "--scheduler", "cosine",
        "--warmup-ratio", "0.05",
        "--cfg-prob", "0.1",
        "--model", "SiT-B/2",
        "--mixed-precision", "bf16",
        "--out-dir", "checkpoints",
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed: {e.returncode}")
        sys.exit(e.returncode)
    
    print()
    print("[INFO] Training finished")


if __name__ == "__main__":
    main()
