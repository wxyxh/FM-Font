# sampling_utils.py
import torch
import torchvision
import numpy as np
from PIL import Image

# 修正：从新的文件名和函数名导入
from sampler import flow_matching_euler_sampler

def create_grid_torchvision(image_list, nrows=2, ncols=4, image_size=224, device="cpu"):
    """
    创建图像网格
    """
    processed_list = []
    for img in image_list:
        if isinstance(img, Image.Image):
            img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
        elif isinstance(img, torch.Tensor):
            img_tensor = img.to(device)
            if img_tensor.ndim == 3 and img_tensor.shape[0] != 3:
                # 处理单通道到三通道的转换
                img_tensor = img_tensor.repeat(3, 1, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        processed_list.append(img_tensor)

    target_num = nrows * ncols
    current_num = len(processed_list)

    if current_num < target_num:
        template = processed_list[0]
        for _ in range(target_num - current_num):
            processed_list.append(torch.ones_like(template, device=device))

    batch = torch.stack(processed_list[:target_num]).to(device)

    grid = torchvision.utils.make_grid(
        batch,
        nrow=ncols,
        padding=2,
        normalize=False,
        pad_value=1.0 
    )

    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    return grid_image


@torch.no_grad()
def sample_hanzi_flow_matching(
    model,
    vae,
    z_glyph,
    z_style,
    device="cuda",
    cfg_scale=3.0,
    latent_size=28, # 对应 224 // 8
):
    """
    使用 Flow Matching 单步采样生成汉字
    """
    model.eval()
    vae.eval()

    # 1. 采样初始噪声 z1 (t=1)
    # 注意：SiT 在 latent 空间通常是 4 通道
    latents = torch.randn((1, 4, latent_size, latent_size), device=device)
    #latents = torch.randn((1, 4, latent_size, latent_size), device=device) * 0.18215

    # 2. 调用重命名后的流匹配采样器
    x0_latent = flow_matching_euler_sampler(
        model=model,
        latents=latents,
        z_glyph=z_glyph,
        z_style=z_style,
        cfg_scale=cfg_scale,
        num_steps=20,
    )

    # 3. VAE 解码 (使用与训练一致的缩放系数)
    # 建议使用 0.18215 或 vae.config.scaling_factor
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    img = vae.decode(x0_latent / scaling_factor).sample

    # [-1, 1] -> [0, 1]
    img = (img.clamp(-1, 1) + 1) / 2
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).round().astype(np.uint8)

    return Image.fromarray(img)


@torch.no_grad()
def generate_hanzi_grid_flow_matching(
    model,
    vae,
    dataset,
    device="cuda",
    font_ids=None,
    style_id=12,
    num_chars=6625,
    cfg_scale=3.0,
):
    """
    生成固定字体和样式的对比网格
    """
    if font_ids is None:
        font_ids = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]

    images = []
    latent_size = model.x_embedder.img_size[0] # 从模型中动态获取 latent 分辨率

    for idx in font_ids:
        # 根据你的 Dataset 索引逻辑计算
        data_index = num_chars * style_id + idx
        try:
            item = dataset[data_index]
            z_g = item["z_glyph"].unsqueeze(0).to(device)
            z_s = item["z_style"].unsqueeze(0).to(device)

            img = sample_hanzi_flow_matching(
                model=model,
                vae=vae,
                z_glyph=z_g,
                z_style=z_s,
                device=device,
                cfg_scale=cfg_scale,
                latent_size=latent_size
            )
            images.append(img)
        except Exception as e:
            print(f"Skipping index {data_index}: {e}")

    # 布局为 2x4
    grid = create_grid_torchvision(images, nrows=2, ncols=4, device=device)
    
    # 确保返回的是 0-255 的 uint8 数组，方便 PIL 直接保存
    if grid.max() <= 1.0:
        grid = (grid * 255).astype(np.uint8)
        
    return images, grid