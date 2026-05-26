"""
Single-scale continuous VAE for LR images.

Ported from myvaex/models/lr_vae.py. In var/ we only ever construct this when
`--stage1_ckpt` is non-empty, and the model is always frozen (`test_mode=True`).
"""
from typing import Tuple

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import DiagonalGaussianDistribution


class LR_VAE(nn.Module):
    """Single-scale continuous VAE for low-resolution images.

    Default config maps a 64x64 (or 80x80) input to a 4x4 (or 5x5) latent via
    16x downsampling.
    """

    def __init__(
        self,
        z_channels: int = 32,
        ch: int = 128,
        dropout: float = 0.0,
        beta: float = 1.0,
        quant_conv_ks: int = 3,
        test_mode: bool = True,
        img_channels: int = 3,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.Cvae = z_channels
        self.kl_weight = beta
        self.img_channels = int(img_channels)

        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=self.img_channels, ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )

        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)  # 16x

        self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.mean_logvar_conv = nn.Conv2d(self.Cvae, 2 * self.Cvae, kernel_size=1, stride=1, padding=0)
        self._init_mean_logvar_conv()

        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    def _init_mean_logvar_conv(self):
        nn.init.xavier_normal_(self.mean_logvar_conv.weight.data, gain=1.0)
        if self.mean_logvar_conv.bias is not None:
            Cvae = self.Cvae
            self.mean_logvar_conv.bias.data[:Cvae].zero_()
            self.mean_logvar_conv.bias.data[Cvae:].fill_(-2.0)

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_encoded = self.quant_conv(self.encoder(inp))
        moments = self.mean_logvar_conv(f_encoded)
        posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
        z = posterior.sample() if self.training else posterior.mode()
        kl_loss = torch.mean(posterior.kl()) * self.kl_weight
        rec = self.decoder(self.post_quant_conv(z))
        return rec, z, kl_loss

    def encode_to_posterior_stats(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_encoded = self.quant_conv(self.encoder(inp))
        moments = self.mean_logvar_conv(f_encoded)
        posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
        return posterior.mean, posterior.logvar

    def encode_to_posterior_mean(self, inp: torch.Tensor) -> torch.Tensor:
        mean, _ = self.encode_to_posterior_stats(inp)
        return mean

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z)).clamp(-1, 1)
