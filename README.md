# **FM-Font: Flow Matching with Disentangled Dual-Branch Encoding for Few-shot Chinese Font Generation**

A Chinese font generation project based on SiT (Scalable diffusion Models with Transformers) architecture, using Flow Matching for training.

## Download Data and Features

Data files and pre-computed features can be downloaded from Baidu NetDisk:

https://pan.baidu.com/s/1fREkNLSgvycYzA6M2QktuQ?pwd=aw2g

This includes:

- `hanzi_font_clip_features.pt`: Pre-computed CLIP features
- Training data (data/fonts/)

## File Structure

```
FM-Font/
├── run.py                  # Training launch script (auto-detects checkpoint to resume)
├── train.py                # Main training script
├── evaluate.py             # Model evaluation script
├── sit.py                  # SiT model architecture
├── loss.py                 # Flow Matching loss function
├── dataset.py              # Dataset loading and sampler
├── sampler.py              # Sampler (Euler method)
├── sampling_utils.py       # Sampling utilities
├── scripts/                # Utility scripts directory
│   └── encode_hanzi_to_moments.py  # VAE encoding script
├── features/               # Pre-computed features directory
│   └── hanzi_font_clip_features.pt
├── data/                   # Data directory
│   └── fonts/
│       ├── train/          # Training set
│       ├── valid/          # Validation set (unseen fonts - seen characters)
│       ├── valid2/         # Validation set (seen fonts - unseen characters)
│       └── valid3/         # Validation set (unseen fonts - unseen characters)
└── checkpoints/            # Model weights output directory
```

## Environment Setup

### Dependencies

```bash
torch>=2.0.0
torchvision
transformers
timm
diffusers
accelerate
pillow
numpy
tqdm
scikit-image
lpips (optional, for LPIPS metric)
pytorch-fid (optional, for FID metric)
```

### Installation

```bash
pip install torch torchvision transformers timm diffusers accelerate pillow numpy tqdm scikit-image lpips pytorch-fid
```

## Usage

### Data Preparation

1. Organize font images with the following structure:

```
fonts/
├── id_0/
│     ├── 00000.png
│     ├── 00001.png
│     └── ...
├── id_1/
│     ├── 00000.png
│     └── ...
└── ...
```

2. Pre-compute CLIP features (need to be generated in advance):

```python
# Use CLIP model to extract glyph and style features
# Save to features/hanzi_font_clip_features.pt
```

3. Pre-compute VAE moments (for latent space):

```bash
# Use pre-trained VAE to encode images to latent moments
python scripts/encode_hanzi_to_moments.py \
    --data-dir data/fonts/train \
    --feature-path features/hanzi_font_clip_features.pt \
    --output-path moments.pt \
    --batch-size 64
```

### Training Model

#### 1. Configure Accelerate (Single GPU)

```bash
accelerate config
```

When prompted, select:
- **Training type**: `No distributed training`
- **Compute environment**: `This machine`
- **Mixed precision**: `bf16`
- **Dynamically**`False`
- **Machine rank**: `0`
- **Number of machines**: `1`
- **GPU devices**: `0` (or your GPU device id)

Or create config file at `~/.cache/huggingface/accelerate/default_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

#### 2. Start Training

**For 24GB VRAM (e.g., RTX 3090, RTX4090):**
```bash
accelerate launch train.py \
    --data-dir fonts/train \
    --feature-path features/hanzi_font_clip_features.pt \
    --moments-path ./fonts/moments.pt \
    --batch-size 128 \
    --grad-accum 2 \
    --epochs 400 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-steps 40000 \
    --cfg-prob 0.1 \
    --model SiT-B/2 \
    --mixed-precision bf16 \
    --out-dir checkpoints
```

**For 48GB VRAM (e.g., A100, RTX 6000 Ada):**
```bash
accelerate launch train.py \
    --data-dir fonts/train \
    --feature-path features/hanzi_font_clip_features.pt \
    --moments-path ./fonts/moments.pt \
    --batch-size 256 \
    --grad-accum 1 \
    --epochs 400 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-steps 40000 \
    --cfg-prob 0.1 \
    --model SiT-B/2 \
    --mixed-precision bf16 \
    --out-dir checkpoints
```

#### 3. Alternative: Use run.py (Auto-resume)

```bash
python run.py
```

This script automatically detects the latest checkpoint and resumes training.


## License

This project is for academic research purposes only.

## References

- [SiT: Scalable diffusion Models with Transformers](https://ieeexplore.ieee.org/document/10377858)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
