# sampler.py
import torch

@torch.no_grad()
def flow_matching_one_step_sampler(
    model,
    latents: torch.Tensor,
    z_glyph: torch.Tensor,
    z_style: torch.Tensor,
    cfg_scale: float = 3.0,
):
    """
    流匹配单步采样器（等价于极简欧拉积分）。
    """
    device = latents.device
    batch_size = latents.shape[0]
    dtype = latents.dtype # ❗ 确保精度一致

    r = torch.zeros(batch_size, device=device, dtype=dtype)
    t = torch.ones(batch_size, device=device, dtype=dtype)

    if cfg_scale > 1.0:
        latents_comb = torch.cat([latents, latents], dim=0)
        r_comb = torch.cat([r, r], dim=0)
        t_comb = torch.cat([t, t], dim=0)
        zg_comb = torch.cat([z_glyph, torch.zeros_like(z_glyph)], dim=0)
        zs_comb = torch.cat([z_style, torch.zeros_like(z_style)], dim=0)

        v_cond, v_uncond = model(
            latents_comb, r_comb, t_comb, z_glyph=zg_comb, z_style=zs_comb
        ).chunk(2, dim=0)

        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
    else:
        v_final = model(latents, r, t, z_glyph=z_glyph, z_style=z_style)

    # SiT 线性路径: z_t = (1-t)x0 + t*x1 -> v = x1 - x0
    # 在 t=1 时, z_1 = x1, 所以 x0 = z_1 - v
    return latents - v_final

@torch.no_grad()
def flow_matching_euler_sampler(
    model,
    latents: torch.Tensor,
    z_glyph: torch.Tensor,
    z_style: torch.Tensor,
    cfg_scale: float = 3.0,
    num_steps: int = 20,
):
    """
    欧拉采样器。沿着 ODE 轨迹 z' = v(z, t) 反向从 t=1 积分到 t=0。
    """
    device = latents.device
    batch_size = latents.shape[0]
    dtype = latents.dtype
    z = latents.clone()
    
    dt = 1.0 / num_steps 
    
    for i in range(num_steps):
        # 当前时间步 t_curr
        t_val = 1.0 - (i * dt)
        t = torch.full((batch_size,), t_val, device=device, dtype=dtype)
        r = torch.zeros_like(t)

        if cfg_scale > 1.0:
            z_comb = torch.cat([z, z], dim=0)
            t_comb = torch.cat([t, t], dim=0)
            r_comb = torch.cat([r, r], dim=0)
            zg_comb = torch.cat([z_glyph, torch.zeros_like(z_glyph)], dim=0)
            zs_comb = torch.cat([z_style, torch.zeros_like(z_style)], dim=0)

            v_cond, v_uncond = model(
                z_comb, r_comb, t_comb, z_glyph=zg_comb, z_style=zs_comb
            ).chunk(2, dim=0)
            v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_final = model(z, r, t, z_glyph=z_glyph, z_style=z_style)

        # ❗ 重要：沿着负方向步进（因为是从 t=1 向 t=0 移动）
        # dz = v * dt_step, 其中 dt_step = - (1.0/num_steps)
        z = z - v_final * dt

    return z