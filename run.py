#!/usr/bin/env python3
"""
Flow Matching 训练启动脚本
"""

import os
import re
import sys
import shutil
import glob
import subprocess

def get_latest_checkpoint_epoch(checkpoint_dir):
    """获取最新的 checkpoint epoch"""
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
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("[INFO] 正在启动 Flow Matching 训练...")
    print()
    
    # 查找 accelerate 命令
    accelerate_cmd = shutil.which("accelerate")
    if accelerate_cmd is None:
        print("[ERROR] 找不到 accelerate 命令")
        print("       请确保已安装: pip install accelerate")
        sys.exit(1)
    
    print(f"[OK] accelerate 命令: {accelerate_cmd}")
    
    # 检查训练脚本是否存在
    train_script = os.path.join("", "train.py")
    if not os.path.exists(train_script):
        print(f"[ERROR] 找不到训练脚本 {train_script}")
        sys.exit(1) 
    
    # 检测最新的 checkpoint
    checkpoint_dir = "checkpoints"
    latest_epoch = get_latest_checkpoint_epoch(checkpoint_dir)
    
    # 训练参数
    target_epochs = 1000  # 目标训练轮数
    
    if latest_epoch > 0:
        # 如果有 checkpoint，从它继续训练
        target_epochs = latest_epoch + target_epochs
        print(f"[INFO] 发现 checkpoint: epoch {latest_epoch}")
        print(f"[INFO] 将从 epoch {latest_epoch} 继续训练到 epoch {target_epochs}")
    else:
        print(f"[INFO] 未找到 checkpoint，将从头开始训练到 epoch {target_epochs}")
    
    print("[INFO] 开始训练...")
    print()
    
    # 训练命令参数
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
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 训练失败，退出码: {e.returncode}")
        sys.exit(e.returncode)
    
    print()
    print("[INFO] 训练结束")


if __name__ == "__main__":
    main()
