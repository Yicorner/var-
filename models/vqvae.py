"""
Continuous multi-scale VAE for var/.

Drop-in replacement of the old discrete VQ-VAE. The class is still called VQVAE for
backward compatibility with the rest of the codebase (and with `train`/`metric.py`
that reference `vae_local: VQVAE`). Internally it now wraps a continuous
`ContinuousMultiScaleQuantizer` and exposes both the legacy interface
(`fhat_to_img`, `embed_to_fhat`, `img_to_reconstructed_img`) and the new
continuous-AR interface (`img_to_ms_continuous_input`).

The VAE is frozen at construction time (`test_mode=True`) and intended to be loaded
from a myvaex stage2 checkpoint.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import (
    ContinuousMultiScaleQuantizer,
    DiagonalGaussianDistribution,
    VectorQuantizer2,                # alias of ContinuousMultiScaleQuantizer; re-exported for legacy `from models.vqvae import VectorQuantizer2`
)


class VQVAE(nn.Module):
    """Continuous multi-scale VAE (legacy name kept for compat)."""

    def __init__(
        self,
        vocab_size: int = 0,      # kept for CLI/ckpt back-compat; unused in continuous mode
        z_channels: int = 32,
        ch: int = 128,
        dropout: float = 0.0,
        beta: float = 1.0,        # KL loss weight (continuous), formerly commitment loss
        using_znorm: bool = False, # kept for CLI/ckpt back-compat; unused
        quant_conv_ks: int = 3,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        default_qresi_counts: int = 0,
        v_patch_nums: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        test_mode: bool = True,
        img_channels: int = 3,
    ):
        super().__init__()
        self.quant_conv_ks = quant_conv_ks
        self.dropout = dropout
        self.ch = ch
        self.test_mode = test_mode
        self.Cvae = z_channels
        self.img_channels = int(img_channels)
        # We keep `V` only as a numerical marker for log lines (some legacy code
        # accesses `vae_local.vocab_size`); continuous VAE has no codebook.
        self.V = int(vocab_size) if vocab_size else 0
        self.vocab_size = self.V

        # The encoder/decoder match myvaex (vq-f16) and stage2 checkpoint layout.
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=self.img_channels, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        ddconfig.pop('double_z', None)
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.quantize: ContinuousMultiScaleQuantizer = ContinuousMultiScaleQuantizer(
            Cvae=self.Cvae, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums,
            quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)

        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    # ===================== `forward` is only used in VAE training, not in var =====================
    def forward(self, inp, ret_usages: bool = False):
        f_hat, usages, kl_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, kl_loss

    def fhat_to_img(self, f_hat: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_f(self, inp_img_no_grad: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(inp_img_no_grad))

    # ===================== continuous AR interface (NEW) =====================
    @torch.no_grad()
    def img_to_ms_continuous_input(
        self, inp_img_no_grad: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """End-to-end helper for SRVAR training:

        HR image -> encoder -> quant_conv -> per-scale posterior means / teacher
        forcing input. Returns:
            ms_h_target: List[Tensor], each `[B, C, pn, pn]`, the DiffLoss target.
            ms_x_input:  `[B, L-1, C]` teacher-forcing input (or `None` if SN==1).
            f_hat_full:  `[B, C, H, W]` final accumulated feature (for debugging).

        Always invoked under `no_grad`; VAE is frozen.
        """
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_var_input_continuous(f)

    def img_to_scale_posterior_stats(self, x: torch.Tensor, scale_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        return self.quantize.get_scale_posterior_stats(f, scale_index=scale_index)

    def img_to_first_scale_posterior_mean(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.img_to_scale_posterior_stats(x, scale_index=0)
        return mean

    # ===================== legacy multi-scale reconstruction helpers =====================
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one: bool = False):
        if last_one:
            return self.decoder(self.post_quant_conv(
                self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)
            )).clamp_(-1, 1)
        return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]

    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one: bool = False):
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_fhat_multiscale(f, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    # ===================== ckpt loading compatibility =====================
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        # Strip legacy VQ-VAE-only keys, mirroring myvaex.
        for key in ('quantize.ema_vocab_hit_SV', 'quantize.embedding.weight', 'quantize.vocab_size'):
            if key in state_dict:
                print(f"[VQVAE.load_state_dict] dropping legacy key: {key}")
                del state_dict[key]
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
