import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vae import Encoder


class Stage3Scale0Encoder(nn.Module):
    """Frozen LR -> stage2 scale[0] latent encoder used by SRVAR."""

    def __init__(
        self,
        z_channels: int = 32,
        ch: int = 128,
        dropout: float = 0.0,
        quant_conv_ks: int = 3,
        latent_size: int = 4,
        img_channels: int = 3,
        freeze_mean_head: bool = True,
    ):
        super().__init__()
        self.Cvae = int(z_channels)
        self.ch = int(ch)
        self.dropout = float(dropout)
        self.quant_conv_ks = int(quant_conv_ks)
        self.latent_size = int(latent_size)
        self.img_channels = int(img_channels)

        ddconfig = dict(
            dropout=self.dropout,
            ch=self.ch,
            z_channels=self.Cvae,
            in_channels=self.img_channels,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        )
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.quant_conv = nn.Conv2d(
            self.Cvae,
            self.Cvae,
            self.quant_conv_ks,
            stride=1,
            padding=self.quant_conv_ks // 2,
        )
        self.mean_logvar_conv = nn.Conv2d(self.Cvae, 2 * self.Cvae, kernel_size=1)
        if freeze_mean_head:
            self.freeze_mean_head()

    def freeze_mean_head(self):
        for p in self.mean_logvar_conv.parameters():
            p.requires_grad_(False)
        self.mean_logvar_conv.eval()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        f = self.quant_conv(self.encoder(inp))
        if f.shape[-2:] != (self.latent_size, self.latent_size):
            f = F.adaptive_avg_pool2d(f, output_size=(self.latent_size, self.latent_size))
        with torch.amp.autocast('cuda', enabled=False):
            moments = self.mean_logvar_conv(f.float())
            mean, _ = torch.chunk(moments, 2, dim=1)
        return mean.to(dtype=f.dtype)

    def extra_repr(self):
        return (
            f'Cvae={self.Cvae}, ch={self.ch}, latent_size={self.latent_size}, '
            f'img_channels={self.img_channels}'
        )
