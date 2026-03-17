import os
import re
import sys
import shutil
import glob
import subprocess

def get_latest_checkpoint_epoch(checkpoint_dir):
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
    
    accelerate_cmd = shutil.which("accelerate")
    if accelerate_cmd is None:
        print("[ERROR] accelerate not found")
        sys.exit(1)
    
    train_script = os.path.join("", "train.py")
    if not os.path.exists(train_script):
        print(f"[ERROR] Train script not found: {train_script}")
        sys.exit(1) 
    
    checkpoint_dir = "checkpoints"
    latest_epoch = get_latest_checkpoint_epoch(checkpoint_dir)
    
    target_epochs = 400
    
    if latest_epoch > 0:
        target_epochs = latest_epoch + target_epochs
    
    cmd = [
        "accelerate", "launch",
        train_script,
        "--data-dir", "fonts/train",
        "--feature-path", "features/hanzi_font_clip_features.pt",
        "--moments-path", "./fonts/moments.pt",
        "--batch-size", "128",
        "--grad-accum", "2",
        "--epochs", str(target_epochs),
        "--lr", "5e-5",
        "--weight-decay", "0.01",
        "--warmup-steps", "40000",
        "--cfg-prob", "0.1",
        "--model", "SiT-B/2",
        "--mixed-precision", "bf16",
        "--out-dir", "checkpoints",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
