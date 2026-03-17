import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
from collections import defaultdict
from PIL import Image

# Gaussian Noise Augmentation
class AddGaussianNoise(object):
    """Add Gaussian noise for data augmentation."""
    def __init__(self, mean=0., std=0.01, p=0.2):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor


def get_transform_train():
    """Get training transform with data augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.GaussianBlur(3, sigma=(0.1, 0.4)),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.005, p=0.1),
        transforms.RandomErasing(
            p=0.1,
            scale=(0.005, 0.015),
            ratio=(0.5, 2.0),
            value=0
        ),
    ])


def get_transform_val():
    """Get validation transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

class HanziFontDataset(Dataset):
    """Chinese font image dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        style_folders = sorted(os.listdir(root_dir))
        self.num_fonts = len(style_folders)

        for style_id, style_name in enumerate(style_folders):
            style_path = os.path.join(root_dir, style_name)

            if not os.path.isdir(style_path):
                continue

            for fname in sorted(os.listdir(style_path)):
                if not fname.endswith(".png"):
                    continue
                glyph_id = int(os.path.splitext(fname)[0])
                img_path = os.path.join(style_path, fname)
                self.samples.append((img_path, glyph_id, style_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, glyph_id, style_id = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "glyph_id": torch.tensor(glyph_id, dtype=torch.long),
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long)
        }

class HanziFontDatasetWithIndex(HanziFontDataset):
    """Extended dataset with glyph_id and style_id index mappings."""

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.glyph2indices = defaultdict(list)
        self.style2indices = defaultdict(list)

        for idx, (_, glyph_id, style_id) in enumerate(self.samples):
            self.glyph2indices[glyph_id].append(idx)
            self.style2indices[style_id].append(idx)

class GlyphStyleBatchSampler(BatchSampler):
    """Batch sampler with fixed glyph and style subsets per batch."""
    def __init__(
        self,
        glyph_style_to_indices,
        glyph_ids,
        style_ids,
        glyphs_per_batch=64,
        styles_per_batch=6,
        batches_per_epoch=1000
    ):
        self.map = glyph_style_to_indices
        self.glyph_ids = list(glyph_ids)
        self.style_ids = list(style_ids)
        self.gpb = glyphs_per_batch
        self.spb = styles_per_batch
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            styles = random.sample(self.style_ids, self.spb)
            glyphs = random.sample(self.glyph_ids, self.gpb)

            batch = []

            for s in styles:
                for g in glyphs:
                    key = (g, s)
                    if key in self.map:
                        batch.append(random.choice(self.map[key]))

            if len(batch) == 0:
                continue
                
            if len(batch) != self.gpb * self.spb:
                pass

            yield batch

    def __len__(self):
        return self.batches_per_epoch

def create_full_cartesian_loader(dataset, m=8, n=16, num_workers=4):
    """Create DataLoader with full cartesian sampling."""
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)
        
    all_glyph_ids = list(dataset.glyph2indices.keys())
    all_style_ids = list(dataset.style2indices.keys())
    
    sampler = GlyphStyleBatchSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        glyphs_per_batch=m,
        styles_per_batch=n
    )
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

class StrictContrastiveBatchSampler(BatchSampler):
    """Full contrastive learning sampler covering all glyph-style combinations."""
    def __init__(self, glyph_style_to_indices, glyph_ids, style_ids, m, n):
        self.gs_map = glyph_style_to_indices
        self.glyph_ids = list(glyph_ids)
        self.style_ids = list(style_ids)
        self.m = m
        self.n = n
        
        self.num_style_groups = len(self.style_ids) // m
        self.num_glyph_groups = len(self.glyph_ids) // n
        
        self.total_batches = self.num_style_groups * self.num_glyph_groups

    def __iter__(self):
        shuffled_styles = list(self.style_ids)
        random.shuffle(shuffled_styles)
        
        for j in range(self.num_style_groups):
            batch_styles = shuffled_styles[j * self.m : (j + 1) * self.m]
            
            shuffled_glyphs = list(self.glyph_ids)
            random.shuffle(shuffled_glyphs)
            
            for i in range(self.num_glyph_groups):
                batch_glyphs = shuffled_glyphs[i * self.n : (i + 1) * self.n]
                
                batch_indices = []
                for s_id in batch_styles:
                    for g_id in batch_glyphs:
                        idx_list = self.gs_map.get((g_id, s_id))
                        if idx_list:
                            batch_indices.append(random.choice(idx_list))
                        else:
                            continue 
                
                if len(batch_indices) > 0:
                    yield batch_indices

    def __len__(self):
        return self.total_batches

def create_strict_contrastive_loader(dataset, m=8, n=16, num_workers=4):
    """Create DataLoader with strict contrastive sampling."""
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)
        
    all_glyph_ids = list(dataset.glyph2indices.keys())
    all_style_ids = list(dataset.style2indices.keys())
    
    sampler = StrictContrastiveBatchSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        m=m,
        n=n
    )
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
#  Diffusion Dataset 

transform_ldm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class HanziFontDiffusionDatasetWithIndexFixed(HanziFontDatasetWithIndex):
    """Diffusion dataset with pre-computed glyph/style features."""

    def __init__(self, root_dir, feature_path, transform=transform_ldm):
        super().__init__(root_dir, transform)

        feat = torch.load(feature_path, map_location="cpu")

        self.z_glyph = feat["z_glyph"]
        self.z_style = feat["z_style"]

    def __getitem__(self, idx):
        img_path, glyph_id, style_id = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "z_glyph": self.z_glyph[idx],
            "z_style": self.z_style[idx],
            "glyph_label": torch.tensor(glyph_id, dtype=torch.long),
            "style_label": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

class HanziFontDiffusionDatasetWithIndex(HanziFontDatasetWithIndex):
    """Conditional diffusion dataset with random glyph/style sampling."""

    def __init__(self, root_dir, feature_path, transform=transform_ldm):
        super().__init__(root_dir, transform)

        feat = torch.load(feature_path, map_location="cpu")

        self.z_glyph = feat["z_glyph"]
        self.z_style = feat["z_style"]

        print(f"z_glyph shape: {self.z_glyph.shape if hasattr(self.z_glyph, 'shape') else len(self.z_glyph)}")
        print(f"z_style shape: {self.z_style.shape if hasattr(self.z_style, 'shape') else len(self.z_style)}")
        print(f"samples length: {len(self.samples)}")
        
        if len(self.z_glyph) != len(self.samples):
            if len(self.z_glyph) > len(self.samples):
                self.z_glyph = self.z_glyph[:len(self.samples)]
            else:
                last_element = self.z_glyph[-1:]
                repeats_needed = len(self.samples) - len(self.z_glyph)
                self.z_glyph = np.concatenate([self.z_glyph, np.tile(last_element, (repeats_needed, 1))])
        
        if len(self.z_style) != len(self.samples):
            if len(self.z_style) > len(self.samples):
                self.z_style = self.z_style[:len(self.samples)]
            else:
                last_element = self.z_style[-1:]
                repeats_needed = len(self.samples) - len(self.z_style)
                self.z_style = np.concatenate([self.z_style, np.tile(last_element, (repeats_needed, 1))])
        
        print(f"After adjustment - z_glyph: {len(self.z_glyph)}, z_style: {len(self.z_style)}, samples: {len(self.samples)}")
        assert len(self.z_glyph) == len(self.samples)
        assert len(self.z_style) == len(self.samples)
    
    def __getitem__(self, idx):
        img_path, glyph_id, style_id = self.samples[idx]

        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        # Random glyph condition
        glyph_indices = self.glyph2indices[glyph_id]
        g_idx = random.choice(glyph_indices)
        z_g = self.z_glyph[g_idx]

        # Random style condition
        style_indices = self.style2indices[style_id]
        s_idx = random.choice(style_indices)
        z_s = self.z_style[s_idx]

        return {
            "image": img,
            "z_glyph": z_g,
            "z_style": z_s,
            "glyph_label": torch.tensor(glyph_id, dtype=torch.long),
            "style_label": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class HanziFontDiffusionDatasetWithIndexAndMoments(
    HanziFontDiffusionDatasetWithIndex
):
    """Diffusion dataset with VAE moments."""

    def __init__(
        self,
        root_dir,
        feature_path,
        moments_path="moments.pt",
        transform=None,
    ):
        super().__init__(
            root_dir=root_dir,
            feature_path=feature_path,
            transform=transform,
        )

        # Load moments
        moments = torch.load(moments_path, map_location="cpu")

        assert "mean" in moments and "logvar" in moments, \
            "moments.pt must contain 'mu' and 'logvar'"

        self.mu_list = moments["mean"]
        self.logvar_list = moments["logvar"]

        print(f"[Moments] mu shape: {self.mu_list.shape}")
        print(f"[Moments] logvar shape: {self.logvar_list.shape}")
        print(f"[Samples] length: {len(self.samples)}")

        # Align lengths
        if len(self.mu_list) != len(self.samples):
            if len(self.mu_list) > len(self.samples):
                self.mu_list = self.mu_list[: len(self.samples)]
                self.logvar_list = self.logvar_list[: len(self.samples)]
            else:
                repeat_n = len(self.samples) - len(self.mu_list)
                self.mu_list = torch.cat(
                    [self.mu_list, self.mu_list[-1:].repeat(repeat_n, 1, 1, 1)],
                    dim=0
                )
                self.logvar_list = torch.cat(
                    [self.logvar_list, self.logvar_list[-1:].repeat(repeat_n, 1, 1, 1)],
                    dim=0
                )

        assert len(self.mu_list) == len(self.samples)
        assert len(self.logvar_list) == len(self.samples)

        print(
            f"[After Align] mu/logvar length = {len(self.mu_list)}"
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        mu = self.mu_list[idx]
        logvar = self.logvar_list[idx]

        moments = torch.cat([mu, logvar], dim=0)

        data.update({
            "moments": moments,
        })

        return data



class ProbabilisticStyleGlyphSampler(BatchSampler):
    """Probabilistic sampler with style and glyph sampling probabilities."""
    def __init__(self, glyph_style_to_indices, glyph_ids, style_ids, batch_size, style_probs=None, glyph_probs=None):
        self.gs_map = glyph_style_to_indices
        self.glyph_ids = list(glyph_ids)
        self.style_ids = list(style_ids)
        self.batch_size = batch_size

        self.style_probs = style_probs
        if self.style_probs is None:
            self.style_probs = [1.0 / len(self.style_ids)] * len(self.style_ids)
        else:
            s = sum(self.style_probs)
            self.style_probs = [p / s for p in self.style_probs]

        self.glyph_probs = glyph_probs
        if self.glyph_probs is None:
            self.glyph_probs = [1.0 / len(self.glyph_ids)] * len(self.glyph_ids)
        else:
            s = sum(self.glyph_probs)
            self.glyph_probs = [p / s for p in self.glyph_probs]

        self.total_batches = (len(self.style_ids) * len(self.glyph_ids)) // self.batch_size

    def __iter__(self):
        generated_batches = 0

        while generated_batches < self.total_batches:
            sampled_styles = random.choices(
                self.style_ids,
                weights=self.style_probs,
                k=len(self.glyph_ids)
            )

            sampled_glyphs = random.choices(
                self.glyph_ids,
                weights=self.glyph_probs,
                k=len(self.glyph_ids)
            )

            for start in range(0, len(self.glyph_ids), self.batch_size):
                if generated_batches >= self.total_batches:
                    break

                batch_styles = sampled_styles[start:start + self.batch_size]
                batch_glyphs = sampled_glyphs[start:start + self.batch_size]

                batch_indices = []
                for s_id, g_id in zip(batch_styles, batch_glyphs):
                    idx_list = self.gs_map.get((g_id, s_id))
                    if idx_list:
                        batch_indices.append(random.choice(idx_list))

                if len(batch_indices) > 0:
                    generated_batches += 1
                    yield batch_indices

    def __len__(self):
        return self.total_batches



def create_probabilistic_style_glyph_loader(
    dataset,
    batch_size,
    num_workers=8,
    style_probs=None,
    glyph_probs=None
):
    """Create DataLoader with probabilistic sampling."""
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)

    all_glyph_ids = sorted(list(dataset.glyph2indices.keys()))
    all_style_ids = sorted(list(dataset.style2indices.keys()))

    sampler = ProbabilisticStyleGlyphSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        batch_size=batch_size,
        style_probs=style_probs,
        glyph_probs=glyph_probs
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )


def compute_style_probs_from_avg_loss(
    font_avg_loss,
    power=0.5,
    eps=1e-8
):
    """Compute style sampling probabilities from average loss."""
    if isinstance(font_avg_loss, dict):
        style_ids = sorted(font_avg_loss.keys())
        losses = torch.tensor(
            [font_avg_loss[i] for i in style_ids],
            dtype=torch.float32
        )
    else:
        if torch.is_tensor(font_avg_loss):
            losses = font_avg_loss.detach().clone().float()
        else:
            losses = torch.tensor(font_avg_loss, dtype=torch.float32)
        
    style_ids = list(range(len(losses)))

    M = losses.numel()

    if M == 0:
        return [], torch.tensor([])

    valid = losses > 0
    if not valid.any():
        probs = torch.ones_like(losses) / M
        return style_ids, probs

    losses = losses.clone()
    losses = losses - losses[valid].min()
    losses = losses / (losses[valid].max() + eps)

    weights = (losses + eps) ** power
    p_loss = weights / weights.sum()

    min_prob = 0.5 / M
    probs = min_prob + 0.5 * p_loss

    probs = probs / probs.sum()

    return style_ids, probs


if __name__ == "__main__":
  
  transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 灰度 → 3 通道
    transforms.ToTensor(),
  ])
  
  dataset = HanziFontDatasetWithIndex(
    root_dir="data/fonts/train/",
    transform=transform_224)
    
  # 实例化 DataLoader
  loader = create_full_cartesian_loader(
    dataset=dataset, 
    m=8, 
    n=64, 
    num_workers=8
  )
  
  def inspect_one_batch(dataloader):
    batch = next(iter(dataloader))

    images = batch['image']
    glyph_ids = batch['glyph_id'].cpu()
    style_ids = batch['style_id'].cpu()

    unique_glyphs = torch.unique(glyph_ids)
    unique_styles = torch.unique(style_ids)

    print("Batch size:", images.size(0))
    print("Unique glyphs:", len(unique_glyphs))
    print("Unique styles:", len(unique_styles))

    # 进一步 sanity check
    from collections import Counter
    print("Glyph frequency (top 5):",
          Counter(glyph_ids.tolist()).most_common(5))
    print("Style frequency:",
          Counter(style_ids.tolist()))

  inspect_one_batch(loader)
  

  from tqdm import tqdm
  from collections import Counter

  def verify_epoch_statistics(dataloader, total_glyphs=6625, total_styles=80):
    glyph_counts = Counter()
    style_counts = Counter()
    total_samples = 0

    print(f"开始扫描完整 Epoch (共 {len(dataloader)} 个 batches)...")
    for batch in tqdm(dataloader):
      g_ids = batch['glyph_id'].tolist()
      s_ids = batch['style_id'].tolist()
        
      glyph_counts.update(g_ids)
      style_counts.update(s_ids)
      total_samples += len(g_ids)

    # 结果分析
    print("\n" + "="*30)
    print("Epoch 统计报告")
    print("="*30)
    
    # 汉字统计
    g_freqs = list(glyph_counts.values())
    print(f"1. 汉字 (Glyphs):")
    print(f"   - 出现的汉字种类: {len(glyph_counts)} (预期: {total_glyphs // dataloader.batch_sampler.n * dataloader.batch_sampler.n})")
    print(f"   - 每个汉字出现的次数: {set(g_freqs)} 次 (预期: {dataloader.batch_sampler.m} 次)")

    # 字体统计
    s_freqs = list(style_counts.values())
    print(f"\n2. 字体 (Styles):")
    print(f"   - 出现的字体种类: {len(style_counts)} (预期: 80)")
    print(f"   - 字体出现次数范围: {min(s_freqs)} ~ {max(s_freqs)} 次")
    print(f"   - 字体平均出现次数: {sum(s_freqs)/len(s_freqs):.2f} 次")
    
    print(f"\n3. 总样本数: {total_samples}")
    print("="*30)

  # 执行验证
  # 假设 m=8, n=16
  train_loader = create_strict_contrastive_loader(
    dataset=dataset, 
    m=8, 
    n=64, 
    num_workers=8
  )
  
  verify_epoch_statistics(train_loader)
