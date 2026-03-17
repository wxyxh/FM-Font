# loss.py
import torch
import numpy as np
import math

class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            time_sampler="logit_normal",
            time_mu=-0.4,
            time_sigma=1.0,
            label_dropout_prob=0.1,
    ):
        self.path_type = path_type
        self.weighting = weighting
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.label_dropout_prob = label_dropout_prob

    def interpolant(self, t_view, images, noises):
        """
        计算插值样本 z_t 和对应的目标速度 v_target。
        关键修正：在 float32 下计算三角函数以保证数值稳定性，避免混合精度问题。
        """
        target_dtype = images.dtype
        target_device = images.device
        
        # 在 float32 下计算以保证 ODE 轨迹精度，特别是三角函数
        t_f32 = t_view.to(device=target_device, dtype=torch.float32)
        img_f32 = images.to(torch.float32)
        noise_f32 = noises.to(torch.float32)

        if self.path_type == "linear":
            # 线性路径：SiT 标准配置
            alpha_t = 1 - t_f32
            sigma_t = t_f32
            d_alpha_t = -torch.ones_like(t_f32)
            d_sigma_t = torch.ones_like(t_f32)
        elif self.path_type == "cosine":
            # 余弦路径：关键修正 - 使用 tensor 类型的 pi 保证类型一致
            pi = torch.tensor(math.pi, device=target_device, dtype=torch.float32)
            half_pi = pi / 2
            freq = t_f32 * half_pi
            alpha_t = torch.cos(freq)
            sigma_t = torch.sin(freq)
            d_alpha_t = -half_pi * torch.sin(freq)
            d_sigma_t = half_pi * torch.cos(freq)
        else:
            raise NotImplementedError(f"Unknown path_type: {self.path_type}")

        # z_t = alpha_t * x_0 + sigma_t * x_1
        z_t = alpha_t * img_f32 + sigma_t * noise_f32
        # v_target = d(alpha_t)/dt * x_0 + d(sigma_t)/dt * x_1
        v_target = d_alpha_t * img_f32 + d_sigma_t * noise_f32
        
        # 转回原始 dtype
        return z_t.to(target_dtype), v_target.to(target_dtype)

    def apply_cfg_dropout(self, model_kwargs):
        """
        同步丢弃 z_glyph 和 z_style，用于训练无条件生成分支。
        关键修正：使用 torch.where 避免 in-place 操作和广播问题。
        """
        if self.label_dropout_prob <= 0:
            return model_kwargs

        # 获取任意一个条件的 Batch Size 和 Device
        cond_tensor = None
        for k in ["z_glyph", "z_style"]:
            if k in model_kwargs and torch.is_tensor(model_kwargs[k]):
                cond_tensor = model_kwargs[k]
                break
        
        if cond_tensor is None:
            return model_kwargs

        batch_size = cond_tensor.shape[0]
        device = cond_tensor.device

        # 生成同步掩码：同一 batch 的 glyph 和 style 要么都留，要么都丢
        mask = torch.rand(batch_size, device=device) < self.label_dropout_prob
        
        new_kwargs = {}
        for k, v in model_kwargs.items():
            if torch.is_tensor(v) and k in ["z_glyph", "z_style"]:
                # 关键修正：使用 torch.where 避免 in-place 索引问题
                # 扩展 mask 以匹配 v 的维度：(B,) -> (B, 1, 1, ...) 
                mask_expanded = mask.view(batch_size, *([1] * (v.ndim - 1)))
                v_dropped = torch.where(mask_expanded, torch.zeros_like(v), v)
                new_kwargs[k] = v_dropped
            else:
                new_kwargs[k] = v
        return new_kwargs

    def __call__(self, model, images, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = images.shape[0]
        device = images.device

        # 1. 采样时间步 t
        t = self.sample_time_steps(batch_size, device)
        noises = torch.randn_like(images)

        # 2. 准备广播维度 (B, 1, 1, 1) 并进行插值
        t_view = t.view(-1, 1, 1, 1)
        z_t, v_target = self.interpolant(t_view, images, noises)

        # 3. 训练时的 CFG Dropout
        if model.training:
            model_kwargs = self.apply_cfg_dropout(model_kwargs)

        # 4. 模型预测
        # SiT 接收 (z_t, r, t, **kwargs)，其中 r 是起始时间步（通常为 0）
        # r=0 表示标准的 Rectified Flow / Flow Matching 起点
        r = torch.zeros_like(t)
        v_pred = model(z_t, r, t, **model_kwargs)

        # 5. 计算损失：对所有空间维度取平均
        error = v_pred - v_target
        loss_mse = torch.mean(error ** 2, dim=list(range(1, error.ndim))) 

        # 6. 权重处理
        if self.weighting == "adaptive":
            # 适配性权重，防止某些阶段 loss 过大
            # 注意：1e-4 是 epsilon，可根据实际 loss 范围调整
            weights = 1.0 / (loss_mse.detach() + 1e-4)
            loss = (weights * loss_mse).mean()
        else:
            # 标准 Flow Matching 采用 Uniform Weighting (1.0)
            loss = loss_mse.mean()

        return loss, loss_mse.mean()

    def sample_time_steps(self, batch_size, device):
        """
        采样策略。Logit-Normal 在汉字生成中表现更好，因为它更关注中间去噪阶段。
        关键修正：直接在目标设备上创建分布，避免跨设备迁移。
        """
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.time_sampler == "logit_normal":
            # 关键修正：直接在目标设备上创建分布参数
            mu = torch.tensor(self.time_mu, device=device, dtype=torch.float32)
            sigma = torch.tensor(self.time_sigma, device=device, dtype=torch.float32)
            m = torch.distributions.Normal(mu, sigma)
            t = torch.sigmoid(m.sample((batch_size,)))
        else:
            raise ValueError(f"Unknown sampler: {self.time_sampler}")
        return t