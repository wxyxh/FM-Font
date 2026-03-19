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

#### Method 1: Use run.py (Recommended)

```bash
python run.py
```

This script automatically:

- Detects the latest checkpoint
- Resumes training from interruption
- Uses `accelerate` for distributed training

#### Method 2: Use train.py Directly

```bash
python train.py \
    --data-dir data/fonts/train \
    --feature-path features/hanzi_font_clip_features.pt \
    --moments-path ./moments.pt \
    --batch-size 128 \
    --epochs 400 \
    --lr 5e-5 \
    --model SiT-B/2 \
    --mixed-precision bf16 \
    --out-dir checkpoints
```

#### Training Parameters

| Parameter             | Default                         | Description                              |
| --------------------- | ------------------------------- | ---------------------------------------- |
| `--data-dir`        | `fonts/train`                 | Training data directory                  |
| `--feature-path`    | `hanzi_font_clip_features.pt` | Pre-computed features path               |
| `--moments-path`    | `moments.pt`                  | VAE moments path                         |
| `--batch-size`      | 128                             | Batch size                               |
| `--epochs`          | 400                             | Number of epochs                         |
| `--lr`              | 5e-5                            | Learning rate                            |
| `--lr-min`          | 1e-7                            | Minimum learning rate (cosine scheduler) |
| `--scheduler`       | `cosine`                      | Learning rate scheduler                  |
| `--warmup-ratio`    | 0.05                            | Warmup ratio                             |
| `--grad-accum`      | 1                               | Gradient accumulation steps              |
| `--cfg-prob`        | 0.1                             | CFG dropout probability                  |
| `--model`           | `SiT-B/2`                     | Model architecture                       |
| `--mixed-precision` | `bf16`                        | Mixed precision mode                     |
| `--out-dir`         | `checkpoints`                 | Output directory                         |

## License

This project is for academic research purposes only.

## References

- [SiT: Scalable diffusion Models with Transformers](https://arxiv.org/abs/2401.08740)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
