import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#Embedding Layers for Timesteps       
class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps to vectors."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#SiT Model

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True
        )

        self.q_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        q = self.q_norm(x)
        k = self.k_norm(context)

        out, _ = self.attn(
            query=q,
            key=k,
            value=context,
            need_weights=False
        )

        return self.gate * out


class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        qk_norm = block_kwargs.get("qk_norm", False)

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm
        )

        # Glyph cross-attention
        self.norm_cross_glyph = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_glyph = CrossAttention(hidden_size, num_heads)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0
        )

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, zg_token):
        """
        x: (B, N, C) - image patches
        c: (B, C) - fused condition (time + style)
        zg_token: (B, 1, C) - glyph token
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # Cross-attention
        x = x + self.cross_attn_glyph(self.norm_cross_glyph(x), zg_token)

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FeatureEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.projector(x)

class ConditionFuser(nn.Module):
    """Fuse timestep with other features."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fuser = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t_emb, feature_emb):
        combined = torch.cat([t_emb, feature_emb], dim=-1)
        return self.fuser(combined)
    
class SiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        zg_dim=512,
        zs_dim=256,
        **block_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.zg_embedder = FeatureEmbedder(zg_dim, hidden_size)
        self.zs_embedder = FeatureEmbedder(zs_dim, hidden_size)
        
        # Fusion layers
        self.style_fuser = ConditionFuser(hidden_size)
        self.glyph_fuser = ConditionFuser(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero initialization
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, r, t, z_glyph=None, z_style=None):
        """Forward pass with glyph and style conditions."""
        x = self.x_embedder(x) + self.pos_embed

        t_emb = self.t_embedder(t)
        zg_emb = self.zg_embedder(z_glyph)
        zs_emb = self.zs_embedder(z_style)

        # Global condition
        c = self.style_fuser(t_emb, zs_emb)
        
        # Spatial guidance
        zg_fused = self.glyph_fuser(t_emb, zg_emb)
        zg_token = zg_fused.unsqueeze(1)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x, c, zg_token)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

#  Positional Embedding Helpers 
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

SiT_models = {
    'SiT-XL/2': lambda **kwargs: SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
    'SiT-L/2':  lambda **kwargs: SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs),
    'SiT-B/2':  lambda **kwargs: SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs),
    'SiT-B/4':  lambda **kwargs: SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)
}
