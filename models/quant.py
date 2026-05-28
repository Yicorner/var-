"""
Continuous multi-scale quantizer for var/.

Ported from myvaex/models/quant.py. The key extension is `f_to_var_input_continuous`
which provides both the per-scale teacher-forcing target (`ms_h_target`, the
deterministic posterior mean at each scale) and the per-scale teacher-forcing input
sequence (`ms_x_input`, equivalent to the discrete `idxBl_to_var_input` for continuous
latents). VAR training consumes both.
"""
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


__all__ = ['ContinuousMultiScaleQuantizer', 'DiagonalGaussianDistribution',
           'Phi', 'PhiShared', 'PhiPartiallyShared', 'PhiNonShared']


class DiagonalGaussianDistribution(object):
    """Diagonal Gaussian distribution for reparameterization trick."""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # Stricter clamp for logvar to prevent var explosion (same as myvaex).
        self.logvar = torch.clamp(self.logvar, -10.0, 5.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                                   dim=[1, 2, 3])
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3])

    def mode(self):
        return self.mean


class ContinuousMultiScaleQuantizer(nn.Module):
    """Continuous multi-scale "quantizer" using per-scale Gaussian posteriors.

    Despite the name we keep for backward compatibility with the old VQ-VAE config,
    there is NO codebook. Each scale predicts a diagonal Gaussian over the residual
    feature map, samples (training) / takes the mode (inference), and accumulates.
    """

    def __init__(
        self, Cvae, beta: float = 1.0,                # beta is kl_weight; kept named beta for compat
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,
        vocab_size=None,                              # kept for cli/ckpt compat, unused
        using_znorm=None,                             # kept for cli/ckpt compat, unused
    ):
        super().__init__()
        self.Cvae: int = Cvae
        self.v_patch_nums: Tuple[int] = tuple(v_patch_nums)
        self.kl_weight: float = beta

        # quant_resi blocks (residual refinement after upsampling), identical to myvaex.
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared([
                (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(default_qresi_counts or len(self.v_patch_nums))
            ])
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([
                (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(share_quant_resi)
            ]))

        # Predict per-scale Gaussian moments (mean | logvar).
        self.mean_logvar_conv = nn.Conv2d(Cvae, 2 * Cvae, kernel_size=1, stride=1, padding=0)
        self._init_mean_logvar_conv()

        # Progressive training placeholder (not supported).
        self.prog_si = -1

    def _init_mean_logvar_conv(self):
        nn.init.xavier_normal_(self.mean_logvar_conv.weight.data, gain=1.0)
        if self.mean_logvar_conv.bias is not None:
            Cvae = self.Cvae
            self.mean_logvar_conv.bias.data[:Cvae].zero_()
            self.mean_logvar_conv.bias.data[Cvae:].fill_(-2.0)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, kl_weight={self.kl_weight}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ===================== utility: per-scale posterior stats (used for alignment / debugging) =====================
    def get_scale_posterior_stats(self, f_BChw: torch.Tensor, scale_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= scale_index < len(self.v_patch_nums)):
            raise IndexError(f'{scale_index=} out of range for {self.v_patch_nums=}')
        pn = self.v_patch_nums[scale_index]
        if scale_index != len(self.v_patch_nums) - 1:
            scale_feature = F.interpolate(f_BChw, size=(pn, pn), mode='area')
        else:
            scale_feature = f_BChw
        moments = self.mean_logvar_conv(scale_feature)
        posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
        return posterior.mean, posterior.logvar

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape

        f_rest = f_BChw.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.amp.autocast('cuda', enabled=False):
            total_kl_loss = 0.0
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums):
                if si != SN - 1:
                    rest_scale = F.interpolate(f_rest, size=(pn, pn), mode='area')
                else:
                    rest_scale = f_rest

                moments = self.mean_logvar_conv(rest_scale)
                posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
                if self.training:
                    h_scale = posterior.sample()
                else:
                    h_scale = posterior.mode()

                kl_loss_scale = posterior.kl()
                B_s, C_s, H_s, W_s = rest_scale.shape
                kl_per_elem = kl_loss_scale / (C_s * H_s * W_s)
                kl_loss_scale = torch.mean(torch.log1p(kl_per_elem))

                if si != SN - 1:
                    h_BChw = F.interpolate(h_scale, size=(H, W), mode='bicubic').contiguous()
                else:
                    h_BChw = h_scale.contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)

                f_hat = f_hat + h_BChw
                f_rest = f_rest - h_BChw
                total_kl_loss = total_kl_loss + kl_loss_scale

            total_kl_loss = total_kl_loss / SN * self.kl_weight

        usages = None
        return f_hat.to(dtype), usages, total_kl_loss
    # ===================== end of `forward` =====================

    # ===================== inference helpers =====================
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw: Union[List[torch.Tensor], torch.Tensor] = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                h_BChw = ms_h_BChw[si]
                if si < SN - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)
        return ls_f_hat_BChw

    def f_to_fhat_multiscale(self, f_BChw: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        B, C, H, W = f_BChw.shape
        f_rest = f_BChw.clone()
        f_hat = torch.zeros_like(f_rest)
        ls_f_hat: List[torch.Tensor] = []
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            if 0 <= self.prog_si < si:
                break
            if si != SN - 1:
                rest_scale = F.interpolate(f_rest, size=(ph, pw), mode='area')
            else:
                rest_scale = f_rest
            moments = self.mean_logvar_conv(rest_scale)
            posterior = DiagonalGaussianDistribution(moments, deterministic=True)
            h_scale = posterior.mode()
            if si != SN - 1:
                h_BChw = F.interpolate(h_scale, size=(H, W), mode='bicubic').contiguous()
            else:
                h_BChw = h_scale.contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat = f_hat + h_BChw
            f_rest = f_rest - h_BChw
            ls_f_hat.append(f_hat.clone())
        return ls_f_hat

    # ===================== NEW: continuous teacher-forcing helpers for SRVAR =====================
    def f_to_var_input_continuous(
        self,
        f_BChw: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Build (ms_h_target, ms_x_input, f_hat_full) for continuous SRVAR training.

        - `ms_h_target[si]`: `[B, C, pn_si, pn_si]`, the per-scale posterior `mode()`
          BEFORE `quant_resi`. This is the DiffLoss target for the tokens of scale `si`.
        - `ms_x_input`: `[B, L-1, C]` teacher-forcing input. For each `si` in `0..SN-2`,
          we downsample the cumulative `f_hat` (after `quant_resi` and accumulation
          UP TO and INCLUDING scale `si`) to `(pn_{si+1}, pn_{si+1})`, flatten and concat.
          Returns `None` if `len(patch_nums) == 1`.
        - `f_hat_full`: `[B, C, H, W]`, the full accumulated feature (useful for sanity).

        The function is called with VAE frozen and `f_BChw` already detached.
        """
        assert f_BChw.dim() == 4, f'expected [B,C,H,W], got {f_BChw.shape}'
        B, C, H, W = f_BChw.shape
        SN = len(self.v_patch_nums)
        f_rest = f_BChw.clone()
        f_hat = torch.zeros_like(f_rest)

        ms_h_target: List[torch.Tensor] = []
        next_scales: List[torch.Tensor] = []

        for si, pn in enumerate(self.v_patch_nums):
            if si != SN - 1:
                rest_scale = F.interpolate(f_rest, size=(pn, pn), mode='area')
            else:
                rest_scale = f_rest

            moments = self.mean_logvar_conv(rest_scale)
            posterior = DiagonalGaussianDistribution(moments, deterministic=True)
            h_scale = posterior.mode()                              # [B, C, pn, pn]
            ms_h_target.append(h_scale.contiguous())

            if si != SN - 1:
                h_BChw = F.interpolate(h_scale, size=(H, W), mode='bicubic').contiguous()
            else:
                h_BChw = h_scale.contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat = f_hat + h_BChw
            f_rest = f_rest - h_BChw

            if si != SN - 1:
                pn_next = self.v_patch_nums[si + 1]
                next_scales.append(
                    F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
                    .view(B, C, -1).transpose(1, 2)
                )

        ms_x_input = torch.cat(next_scales, dim=1) if len(next_scales) else None
        return ms_h_target, ms_x_input, f_hat

    # ===================== inference: next-scale autoregressive helper =====================
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            f_hat.add_(h)
            pn_next = self.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


# Backward-compat alias so legacy import paths still resolve. We re-export the new
# class under the old name to avoid sprinkling rename edits across the codebase.
VectorQuantizer2 = ContinuousMultiScaleQuantizer
