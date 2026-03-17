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
        """Compute interpolant and target velocity."""
        target_dtype = images.dtype
        target_device = images.device
        
        t_f32 = t_view.to(device=target_device, dtype=torch.float32)
        img_f32 = images.to(torch.float32)
        noise_f32 = noises.to(torch.float32)

        if self.path_type == "linear":
            alpha_t = 1 - t_f32
            sigma_t = t_f32
            d_alpha_t = -torch.ones_like(t_f32)
            d_sigma_t = torch.ones_like(t_f32)
        elif self.path_type == "cosine":
            pi = torch.tensor(math.pi, device=target_device, dtype=torch.float32)
            half_pi = pi / 2
            freq = t_f32 * half_pi
            alpha_t = torch.cos(freq)
            sigma_t = torch.sin(freq)
            d_alpha_t = -half_pi * torch.sin(freq)
            d_sigma_t = half_pi * torch.cos(freq)
        else:
            raise NotImplementedError(f"Unknown path_type: {self.path_type}")

        z_t = alpha_t * img_f32 + sigma_t * noise_f32
        v_target = d_alpha_t * img_f32 + d_sigma_t * noise_f32
        
        return z_t.to(target_dtype), v_target.to(target_dtype)

    def apply_cfg_dropout(self, model_kwargs):
        """Apply CFG dropout."""
        if self.label_dropout_prob <= 0:
            return model_kwargs

        cond_tensor = None
        for k in ["z_glyph", "z_style"]:
            if k in model_kwargs and torch.is_tensor(model_kwargs[k]):
                cond_tensor = model_kwargs[k]
                break
        
        if cond_tensor is None:
            return model_kwargs

        batch_size = cond_tensor.shape[0]
        device = cond_tensor.device

        mask = torch.rand(batch_size, device=device) < self.label_dropout_prob
        
        new_kwargs = {}
        for k, v in model_kwargs.items():
            if torch.is_tensor(v) and k in ["z_glyph", "z_style"]:
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

        # Sample timesteps
        t = self.sample_time_steps(batch_size, device)
        noises = torch.randn_like(images)

        t_view = t.view(-1, 1, 1, 1)
        z_t, v_target = self.interpolant(t_view, images, noises)

        # CFG dropout during training
        if model.training:
            model_kwargs = self.apply_cfg_dropout(model_kwargs)

        r = torch.zeros_like(t)
        v_pred = model(z_t, r, t, **model_kwargs)

        # Compute loss
        error = v_pred - v_target
        loss_mse = torch.mean(error ** 2, dim=list(range(1, error.ndim))) 

        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mse.detach() + 1e-4)
            loss = (weights * loss_mse).mean()
        else:
            loss = loss_mse.mean()

        return loss, loss_mse.mean()

    def sample_time_steps(self, batch_size, device):
        """Sample timesteps."""
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.time_sampler == "logit_normal":
            mu = torch.tensor(self.time_mu, device=device, dtype=torch.float32)
            sigma = torch.tensor(self.time_sigma, device=device, dtype=torch.float32)
            m = torch.distributions.Normal(mu, sigma)
            t = torch.sigmoid(m.sample((batch_size,)))
        else:
            raise ValueError(f"Unknown sampler: {self.time_sampler}")
        return t