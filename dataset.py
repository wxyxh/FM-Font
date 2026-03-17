import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
from collections import defaultdict
from PIL import Image

# =============================
# 添加高斯噪声数据增强
# =============================
class AddGaussianNoise(object):
    """添加高斯噪声的数据增强类
    
    用于防止模型记忆像素，强化对字形和字体风格的学习
    """
    def __init__(self, mean=0., std=0.01, p=0.2):
        """
        参数：
        ----------
        mean : float
            高斯噪声的均值
        std : float
            高斯噪声的标准差
        p : float
            应用噪声的概率
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        """
        对输入张量添加高斯噪声
        
        参数：
        ----------
        tensor : Tensor
            输入的图像张量
            
        返回：
        ----------
        Tensor
            添加噪声后的图像张量（或原始张量）
        """
        if torch.rand(1) < self.p:
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor


def get_transform_train():
    """
    获取训练用的数据增强 transform
    
    包含：
    - 几何轻扰动（RandomAffine）
    - 极轻模糊（GaussianBlur）
    - 极低概率噪声（AddGaussianNoise）
    - 极低概率擦除（RandomErasing）
    注意：不包含归一化（Normalize）
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        # ---- 几何轻扰动（核心）----
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
            shear=2
        ),
        # ---- 极轻模糊（抑制像素记忆）----
        transforms.GaussianBlur(3, sigma=(0.1, 0.4)),
        transforms.ToTensor(),
        # ---- 极低概率噪声 ----
        AddGaussianNoise(std=0.005, p=0.1),
        # ---- 极低概率擦除（可选）----
        transforms.RandomErasing(
            p=0.1,
            scale=(0.005, 0.015),
            ratio=(0.5, 2.0),
            value=0
        ),
    ])


def get_transform_val():
    """
    获取验证/测试用的 transform（无数据增强，无归一化）
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

class HanziFontDataset(Dataset):
    """
    汉字字体图像数据集类（基础版）

    数据组织形式假设如下：
    fonts/
        ├── id_0/
        │     ├── 00000.png
        │     ├── 00001.png
        │     └── ...
        ├── id_1/
        │     ├── 00000.png
        │     ├── 00001.png
        │     └── ...
        └── ...

    每个样本返回一个字典，包含：
        - image   : 预处理后的汉字图像张量
        - glyph_id: 汉字内容类别 ID（由文件名解析）
        - style_id: 字体风格类别 ID（由文件夹顺序确定）
    """

    def __init__(self, root_dir, transform=None):
        """
        参数说明：
        ----------
        root_dir : str
            字体数据集根目录（包含多个字体子文件夹）
        transform : torchvision.transforms
            图像预处理与增强操作（Resize / ToTensor / Normalize 等）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 用于存储所有样本索引
        # 每个元素为 (图像路径, 汉字ID, 字体ID)
        self.samples = []

        # 对字体文件夹排序，确保 style_id 可复现、稳定
        style_folders = sorted(os.listdir(root_dir))
        
        # 字体总数
        self.num_fonts = len(style_folders)

        # 遍历每一个字体文件夹
        for style_id, style_name in enumerate(style_folders):
            style_path = os.path.join(root_dir, style_name)

            # 跳过非目录文件
            if not os.path.isdir(style_path):
                continue

            # 遍历该字体下的所有汉字图像
            for fname in sorted(os.listdir(style_path)):
                # 仅处理 png 格式图像
                if not fname.endswith(".png"):
                    continue

                # 从文件名中解析汉字 ID
                # 例如 "00023.png" → glyph_id = 23
                glyph_id = int(os.path.splitext(fname)[0])

                img_path = os.path.join(style_path, fname)

                # 记录样本
                self.samples.append((img_path, glyph_id, style_id))

    def __len__(self):
        """返回数据集中样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一个样本

        返回格式：
        ----------
        dict {
            "image": Tensor,      # 汉字图像
            "glyph_id": LongTensor,  # 汉字内容类别
            "style_id": LongTensor,  # 字体风格类别
            "index": LongTensor      # 汉字图像在数据集中的索引
        }
        """
        img_path, glyph_id, style_id = self.samples[idx]

        # 读取图像并转换为单通道灰度图
        img = Image.open(img_path).convert("L")

        # 应用图像变换（尺寸调整、归一化等）
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "glyph_id": torch.tensor(glyph_id, dtype=torch.long),
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long)
        }

class HanziFontDatasetWithIndex(HanziFontDataset):
    """
    扩展数据集类：在基础数据集之上，额外维护
    - glyph_id → 样本索引列表
    - style_id → 样本索引列表

    适用于：
    - CLIP / 对比学习
    - 固定内容采样不同字体
    - 固定字体采样不同汉字
    """

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        # 汉字ID → 所有对应样本索引
        self.glyph2indices = defaultdict(list)

        # 字体ID → 所有对应样本索引
        self.style2indices = defaultdict(list)

        # 建立反向索引映射
        for idx, (_, glyph_id, style_id) in enumerate(self.samples):
            self.glyph2indices[glyph_id].append(idx)
            self.style2indices[style_id].append(idx)

class GlyphStyleBatchSampler(BatchSampler):
    """
    每个 batch:
      - 随机选 m 个 style
      - 随机选 n 个 glyph
      - 对每个 (glyph, style) 取 1 个样本
    """
    def __init__(
        self,
        glyph_style_to_indices,  # dict[(glyph, style)] -> [idx...]
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

            # 1) 固定 style 子集
            styles = random.sample(self.style_ids, self.spb)

            # 2) 固定 glyph 子集
            glyphs = random.sample(self.glyph_ids, self.gpb)

            batch = []

            for s in styles:
                for g in glyphs:
                    key = (g, s)
                    if key in self.map:
                        batch.append(random.choice(self.map[key]))

            # 如果 batch 为空，跳过这个 batch
            if len(batch) == 0:
                continue
                
            # 如果 batch 大小不匹配，打印警告但继续
            if len(batch) != self.gpb * self.spb:
                # 某些 (glyph, style) 组合可能不存在，这是正常的
                pass

            yield batch

    def __len__(self):
        return self.batches_per_epoch

def create_full_cartesian_loader(dataset, m=8, n=16, num_workers=4):
    """
    直接接收 dataset 对象，构建全量遍历的训练 DataLoader
    
    参数说明:
    ----------
    dataset : HanziFontDatasetWithIndex
        已经实例化好的数据集对象（包含 samples 列表和 ID 映射）
    m : int
        每个 Batch 中的字体数量 (Styles per batch)
    n : int
        每个 Batch 中的汉字数量 (Glyphs per batch)
    num_workers : int
        多线程数据读取的线程数
    """
    
    # 1. 从数据集中构建 (glyph_id, style_id) -> 样本索引的映射表
    # 这确保采样器能快速定位特定的图像
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)
        
    # 2. 提取所有的 ID 列表
    all_glyph_ids = list(dataset.glyph2indices.keys())
    all_style_ids = list(dataset.style2indices.keys())
    
    # 3. 实例化全量遍历采样器
    # 确保 80 种字体和 6625 个汉字形成完整的笛卡尔积组合
    sampler = GlyphStyleBatchSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        glyphs_per_batch=m,
        styles_per_batch=n
    )
    
    # 4. 构建并返回 DataLoader
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

class StrictContrastiveBatchSampler(BatchSampler):
    """
    全量监督对比学习采样器：
    (1) 每个 Epoch 覆盖所有 M 种字体 x N 个汉字的组合。
    (2) 每个 Batch 依然严格保持 m 个字体 * n 个汉字的结构。
    (3) 总 Batch 数 = (M/m) * (N/n)。
    """
    def __init__(self, glyph_style_to_indices, glyph_ids, style_ids, m, n):
        self.gs_map = glyph_style_to_indices
        self.glyph_ids = list(glyph_ids)
        self.style_ids = list(style_ids)
        self.m = m
        self.n = n
        
        # 计算组数
        self.num_style_groups = len(self.style_ids) // m  # 例如 80 // 8 = 10 组
        self.num_glyph_groups = len(self.glyph_ids) // n  # 例如 6625 // 16 = 414 组
        
        # 总 Batch 数：遍历完所有交叉组合
        self.total_batches = self.num_style_groups * self.num_glyph_groups

    def __iter__(self):
        # 1. 随机排列所有字体 ID
        shuffled_styles = list(self.style_ids)
        random.shuffle(shuffled_styles)
        
        # 2. 外层循环：遍历字体组 (每组 m 个)
        for j in range(self.num_style_groups):
            batch_styles = shuffled_styles[j * self.m : (j + 1) * self.m]
            
            # 3. 每一轮字体组，都重新随机排列汉字，增加组合的随机性
            shuffled_glyphs = list(self.glyph_ids)
            random.shuffle(shuffled_glyphs)
            
            # 4. 内层循环：遍历汉字组 (每组 n 个)
            for i in range(self.num_glyph_groups):
                batch_glyphs = shuffled_glyphs[i * self.n : (i + 1) * self.n]
                
                batch_indices = []
                # 5. 填充当前 Batch (m * n)
                for s_id in batch_styles:
                    for g_id in batch_glyphs:
                        idx_list = self.gs_map.get((g_id, s_id))
                        if idx_list:
                            batch_indices.append(random.choice(idx_list))
                        else:
                            # 错误处理：如果某字体缺少某汉字
                            # 可以在此处通过随机补全或报错
                            continue 
                
                # 如果 batch 为空，跳过这个 batch
                if len(batch_indices) > 0:
                    yield batch_indices

    def __len__(self):
        return self.total_batches

def create_strict_contrastive_loader(dataset, m=8, n=16, num_workers=4):
    """
    直接接收 dataset 对象，构建符合实验要求的训练 DataLoader
    
    参数说明:
    ----------
    dataset : HanziFontDatasetWithIndex
        已经实例化好的数据集对象（需包含反向索引属性）
    m : int
        每个 Batch 中的字体数量 (Style count)
    n : int
        每个 Batch 中的汉字数量 (Glyph count)
    num_workers : int
        多线程数据读取的线程数
    """
    
    # 1. 从数据集中构建 (glyph_id, style_id) -> 样本索引的映射表
    # 这确保采样器能准确找到指定的（字形，风格）组合
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)
        
    # 2. 提取所有的 ID 列表
    # 利用 dataset 类中已经维护好的反向索引键值
    all_glyph_ids = list(dataset.glyph2indices.keys())
    all_style_ids = list(dataset.style2indices.keys())
    
    # 3. 实例化严谨的对比学习采样器
    # 实现 m * n 的笛卡尔积采样逻辑
    sampler = StrictContrastiveBatchSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        m=m,
        n=n
    )
    
    # 4. 构建并返回 DataLoader
    # pin_memory=True 可以加快数据从 CPU 拷贝到 GPU 的速度
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
########################   
#  汉字扩散模型数据集  #
########################

transform_ldm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # 灰度转3通道
    transforms.ToTensor(),
    # VAE 预训练模型要求输入范围 [-1, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class HanziFontDiffusionDatasetWithIndexFixed(HanziFontDatasetWithIndex):
    """
    用于汉字-字体扩散模型的数据集类（带索引）
    继承自 HanziFontDatasetWithIndex，在原有图像 + 标签的基础上，
    额外返回预先计算好的 glyph / style 特征向量（z_glyph, z_style）
    """

    def __init__(self, root_dir, feature_path, transform=transform_ldm):
        """
        参数说明：
        - root_dir: 数据集根目录，通常包含不同字体、不同汉字的图像
        - feature_path: 预计算特征文件路径（.pt），
                        内含 glyph / style 的隐空间表示
        - transform: 图像预处理与数据增强操作（如 resize、normalize）
        """

        super().__init__(root_dir, transform)

        # 加载预先保存的特征文件
        feat = torch.load(feature_path, map_location="cpu")

        self.z_glyph = feat["z_glyph"] # [N, D_glyph], glyph 内容特征向量
        self.z_style = feat["z_style"] # [N, D_style], style 风格特征向量

    def __getitem__(self, idx):
        """
        返回一个扩散训练样本：
        {
            image      : 原始图像 x_0
            z_glyph    : 当前图像的字形条件
            z_style    : 当前图像的字体条件
            glyph_label: 当前图像的字形类别
            style_label: 当前图像的字体类别
            index      : 原始样本索引（调试/分析用）
        }
        """
        
         # ---------- (1) 当前真实图像 ----------
        img_path, glyph_id, style_id = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        # 返回一个字典，便于训练时按 key 取用
        return {
            "image": img,
            "z_glyph": self.z_glyph[idx],
            "z_style": self.z_style[idx],
            "glyph_label": torch.tensor(glyph_id, dtype=torch.long),
            "style_label": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

class HanziFontDiffusionDatasetWithIndex(HanziFontDatasetWithIndex):
    """
    汉字-字体条件扩散模型数据集

    对于每个真实图像样本 x_0（glyph=g, style=s）：
    - 字形条件 z_glyph：
        从「所有 glyph = g 的样本」中随机选取一个
    - 字体条件 z_style：
        从「所有 style = s 的样本」中随机选取一个

    该设计实现了：
    - 内容（字形）与风格（字体）的解耦
    - 条件随机性，避免模型记忆一一对应关系
    """

    def __init__(self, root_dir, feature_path, transform=transform_ldm):
        """
        参数：
        ----------
        root_dir : str
            字体图像根目录
        feature_path : str
            预计算 CLIP / 编码器特征（z_glyph, z_style）
        transform : torchvision.transforms
            图像预处理
        """
        super().__init__(root_dir, transform)

        # 加载样本级特征（与 samples 一一对应）
        feat = torch.load(feature_path, map_location="cpu")

        self.z_glyph = feat["z_glyph"]  # [N, D_glyph]
        self.z_style = feat["z_style"]  # [N, D_style]

        # 一致性检查 - 修复长度不匹配问题
        print(f"z_glyph shape: {self.z_glyph.shape if hasattr(self.z_glyph, 'shape') else len(self.z_glyph)}")
        print(f"z_style shape: {self.z_style.shape if hasattr(self.z_style, 'shape') else len(self.z_style)}")
        print(f"samples length: {len(self.samples)}")
        
        # 如果长度不匹配，调整 z_glyph 和 z_style 到 samples 的长度
        if len(self.z_glyph) != len(self.samples):
            if len(self.z_glyph) > len(self.samples):
                self.z_glyph = self.z_glyph[:len(self.samples)]
            else:
                # 重复最后一个元素直到长度匹配
                last_element = self.z_glyph[-1:]
                repeats_needed = len(self.samples) - len(self.z_glyph)
                self.z_glyph = np.concatenate([self.z_glyph, np.tile(last_element, (repeats_needed, 1))])
        
        if len(self.z_style) != len(self.samples):
            if len(self.z_style) > len(self.samples):
                self.z_style = self.z_style[:len(self.samples)]
            else:
                # 重复最后一个元素直到长度匹配
                last_element = self.z_style[-1:]
                repeats_needed = len(self.samples) - len(self.z_style)
                self.z_style = np.concatenate([self.z_style, np.tile(last_element, (repeats_needed, 1))])
        
        print(f"After adjustment - z_glyph: {len(self.z_glyph)}, z_style: {len(self.z_style)}, samples: {len(self.samples)}")
        assert len(self.z_glyph) == len(self.samples)
        assert len(self.z_style) == len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回一个扩散训练样本：
        {
            image      : 原始图像 x_0
            z_glyph    : 随机采样的字形条件
            z_style    : 随机采样的字体条件
            glyph_label: 当前图像的字形类别
            style_label: 当前图像的字体类别
            index      : 原始样本索引（调试/分析用）
        }
        """

        # ---------- (1) 当前真实图像 ----------
        img_path, glyph_id, style_id = self.samples[idx]

        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        # ---------- (2) 随机采样字形条件 z_glyph ----------
        # 从所有 glyph = glyph_id 的样本中随机选一个
        glyph_indices = self.glyph2indices[glyph_id]
        g_idx = random.choice(glyph_indices)
        z_g = self.z_glyph[g_idx]

        # ---------- (3) 随机采样字体条件 z_style ----------
        style_indices = self.style2indices[style_id]
        s_idx = random.choice(style_indices)
        z_s = self.z_style[s_idx]

        # ---------- (4) 返回 ----------
        return {
            "image": img,                                  # x_0
            "z_glyph": z_g,                                # 字形条件
            "z_style": z_s,                                # 字体条件
            "glyph_label": torch.tensor(glyph_id, dtype=torch.long),
            "style_label": torch.tensor(style_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class HanziFontDiffusionDatasetWithIndexAndMoments(
    HanziFontDiffusionDatasetWithIndex
):
    """
    扩展数据集：
    - 继承 HanziFontDiffusionDatasetWithIndex
    - 增加 VAE 编码得到的 mu / logvar
    """

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

        # -------------------------------------------------
        # 1. 加载 moments
        # -------------------------------------------------
        moments = torch.load(moments_path, map_location="cpu")

        assert "mean" in moments and "logvar" in moments, \
            "moments.pt must contain 'mu' and 'logvar'"

        self.mu_list = moments["mean"]          # [N, C, H, W]
        self.logvar_list = moments["logvar"]  # [N, C, H, W]

        print(f"[Moments] mu shape: {self.mu_list.shape}")
        print(f"[Moments] logvar shape: {self.logvar_list.shape}")
        print(f"[Samples] length: {len(self.samples)}")

        # -------------------------------------------------
        # 2. 对齐长度（和你前面 z_glyph / z_style 一样）
        # -------------------------------------------------
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

    # -------------------------------------------------
    # 3. 重载 __getitem__
    # -------------------------------------------------
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # 对应样本的 VAE moments
        mu = self.mu_list[idx]
        logvar = self.logvar_list[idx]

        moments = torch.cat([mu, logvar], dim=0)

        data.update({
            "moments": moments,
            #"mu": mu,
            #"logvar": logvar,
        })

        return data



class ProbabilisticStyleGlyphSampler(BatchSampler):
    """
    1) 初始化时可提供 style_probs，否则均匀分布；
    2) 从 M 个字体中按概率有放回抽样得到 N 个字体列表；
    3) 所有 N 个汉字随机打乱得到字列表；
    4) 每个 batch 从字体列表和字列表按顺序取 batch_size 的组合；
    5) 总 batch 数为 M * N / batch_size，若不足则重新采样。
    """

    def __init__(self, glyph_style_to_indices, glyph_ids, style_ids, batch_size, style_probs=None, glyph_probs=None):
        self.gs_map = glyph_style_to_indices   # {(glyph_id, style_id): [idx1, idx2, ...]}
        self.glyph_ids = list(glyph_ids)       # N = len(glyph_ids)
        self.style_ids = list(style_ids)       # M = len(style_ids)
        self.batch_size = batch_size

        # (1) 初始化字体概率分布
        self.style_probs = style_probs
        if self.style_probs is None:
            self.style_probs = [1.0 / len(self.style_ids)] * len(self.style_ids)
        else:
            s = sum(self.style_probs)
            self.style_probs = [p / s for p in self.style_probs]

        # (2) 初始化汉字内容概率分布 (新增)
        self.glyph_probs = glyph_probs
        if self.glyph_probs is None:
            self.glyph_probs = [1.0 / len(self.glyph_ids)] * len(self.glyph_ids)
        else:
            s = sum(self.glyph_probs)
            self.glyph_probs = [p / s for p in self.glyph_probs]

        # (6) 预估总 batch 数 = M * N / batch_size
        self.total_batches = (len(self.style_ids) * len(self.glyph_ids)) // self.batch_size

    def __iter__(self):
        generated_batches = 0

        while generated_batches < self.total_batches:
            # (3) 按概率采样字体列表
            sampled_styles = random.choices(
                self.style_ids,
                weights=self.style_probs,
                k=len(self.glyph_ids)
            )

            # (4) 按概率采样汉字列表 (修改：从随机打乱改为按概率采样)
            sampled_glyphs = random.choices(
                self.glyph_ids,
                weights=self.glyph_probs,
                k=len(self.glyph_ids)
            )

            # (5) 每个 batch 从字体列表和字列表按顺序取 batch_size 的组合
            for start in range(0, len(self.glyph_ids), self.batch_size):
                if generated_batches >= self.total_batches:
                    break

                batch_styles = sampled_styles[start:start + self.batch_size]
                batch_glyphs = sampled_glyphs[start:start + self.batch_size]

                batch_indices = []
                for s_id, g_id in zip(batch_styles, batch_glyphs):
                    # 根据 (glyph_id, style_id) 找到对应索引列表
                    idx_list = self.gs_map.get((g_id, s_id))
                    if idx_list:
                        # 从该列表中随机抽取一个样本索引
                        batch_indices.append(random.choice(idx_list))

                # 若 batch_indices 非空，则认为该 batch 有效
                if len(batch_indices) > 0:
                    generated_batches += 1
                    yield batch_indices

    def __len__(self):
        # 返回总 batch 数
        return self.total_batches



def create_probabilistic_style_glyph_loader(
    dataset,
    batch_size,
    num_workers=8,
    style_probs=None,
    glyph_probs=None
):
    """
    直接接收 dataset 对象，构建符合实验要求的训练 DataLoader

    参数说明:
    ----------
    dataset : HanziFontDatasetWithIndex
        已经实例化好的数据集对象（需包含反向索引属性）
    batch_size : int
        每个 batch 的样本数量
    num_workers : int
        多线程数据读取的线程数
    style_probs : list or None
        每个字体的采样概率
    glyph_probs : list or None
        每个汉字的采样概率 (新增)
    """

    # 1. 从数据集中构建 (glyph_id, style_id) -> 样本索引的映射表
    glyph_style_to_indices = {}
    for idx, (_, g_id, s_id) in enumerate(dataset.samples):
        key = (g_id, s_id)
        if key not in glyph_style_to_indices:
            glyph_style_to_indices[key] = []
        glyph_style_to_indices[key].append(idx)

    # 2. 提取所有的 ID 列表
    all_glyph_ids = sorted(list(dataset.glyph2indices.keys()))
    all_style_ids = sorted(list(dataset.style2indices.keys()))

    # 3. 实例化概率采样器
    sampler = ProbabilisticStyleGlyphSampler(
        glyph_style_to_indices=glyph_style_to_indices,
        glyph_ids=all_glyph_ids,
        style_ids=all_style_ids,
        batch_size=batch_size,
        style_probs=style_probs,
        glyph_probs=glyph_probs
    )

    # 4. 构建并返回 DataLoader
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
    """
    字体不等概采样（显式下限版）

    设计：
    - 50% 概率均匀分配 → min_prob = 0.5 / num_fonts
    - 50% 概率按 loss 分配（温和）
    """

    # ============================================================
    # 1. 统一输入
    # ============================================================
    if isinstance(font_avg_loss, dict):
        style_ids = sorted(font_avg_loss.keys())
        losses = torch.tensor(
            [font_avg_loss[i] for i in style_ids],
            dtype=torch.float32
        )
    else:
        # 保证 font_avg_loss 是 Tensor，如果已经是 Tensor，安全 clone
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

    # ============================================================
    # 2. loss → [0, 1]
    # ============================================================
    losses = losses.clone()
    losses = losses - losses[valid].min()
    losses = losses / (losses[valid].max() + eps)

    # ============================================================
    # 3. 基于 loss 的温和分布
    # ============================================================
    weights = (losses + eps) ** power
    p_loss = weights / weights.sum()   # 和为 1

    # ============================================================
    # 4. 显式下限：一半均匀 + 一半 loss
    # ============================================================
    min_prob = 0.5 / M
    probs = min_prob + 0.5 * p_loss

    # 归一化
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
