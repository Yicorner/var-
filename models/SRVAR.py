import math
import random
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_vae import Encoder
from models.flex_attn import FlexAttn

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from models.basic import CrossAttnBlock,flash_attn_func, FastRMSNorm, SelfAttnBlock, flash_fused_op_installed, CrossAttention, precompute_rope2d_freqs_grid

from models.diffusion.diffloss import DiffLoss

try:
    from models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

def gather_by_indices(X, Y):
    # 创建批次索引
    batch_indices = torch.arange(X.shape[0], device=X.device).view(-1, 1).expand(-1, Y.shape[1])
    result = X[batch_indices, Y]
    return result

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        self.head_dim = 8
        # if D > 4096:
        #     self.head_dim = 64 
        # else:
        #     self.head_dim = 128
        print(f"TextAttentivePool: Ct5={Ct5}, D={D}, head_dim={self.head_dim}")
        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv): 
        return self.ca(None, ca_kv).squeeze(1)


def _valid_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedLRDetailBranch(nn.Module):
    """A lightweight LR-specific residual branch aligned to the VAE latent grid."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 128):
        super().__init__()
        width = int(max(width, out_channels))
        self.net = nn.Sequential(
            ConvNormAct(in_channels, width, stride=1),
            ConvNormAct(width, width, stride=2),
            ConvNormAct(width, width, stride=2),
            ConvNormAct(width, width * 2, stride=2),
            ConvNormAct(width * 2, width * 2, stride=2),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        detail = self.net(x)
        if detail.shape[-2:] != target_hw:
            detail = F.interpolate(detail, size=target_hw, mode='bilinear', align_corners=False)
        return detail

    def reset_output_to_zero(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


class LearnedLREncoder(nn.Module):
    """Trainable LR encoder with VAE-aligned latent output plus a detail residual."""

    def __init__(self, ddconfig: dict, quant_conv_ks: int, detail_width: int = 128):
        super().__init__()
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.quant_conv = nn.Conv2d(
            ddconfig['z_channels'], ddconfig['z_channels'],
            quant_conv_ks, stride=1, padding=quant_conv_ks // 2,
        )
        self.detail_branch = LearnedLRDetailBranch(
            in_channels=ddconfig['in_channels'],
            out_channels=ddconfig['z_channels'],
            width=detail_width,
        )
        self.fuse = nn.Sequential(
            nn.GroupNorm(_valid_group_count(ddconfig['z_channels']), ddconfig['z_channels']),
            nn.SiLU(inplace=True),
            nn.Conv2d(ddconfig['z_channels'], ddconfig['z_channels'], kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.quant_conv(self.encoder(x))
        detail = self.detail_branch(x, base.shape[-2:])
        mixed = base + detail
        return mixed + self.fuse(mixed)

    def init_from_vae(self, vae_local: VQVAE):
        self.encoder.load_state_dict(vae_local.encoder.state_dict())
        self.quant_conv.load_state_dict(vae_local.quant_conv.state_dict())
        self.reset_residual_to_zero()

    def reset_residual_to_zero(self):
        self.detail_branch.reset_output_to_zero()
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h



class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class SRVAR(nn.Module):
    def __init__(
        self, vae_local,
        low_channel=0, low_len=0,           # low-level-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm (head_aln kept for ckpt compat, ignored)
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,                       # kept for ckpt compat; new continuous head doesn't use it
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify=0,
        inference_mode=False,
        use_ref=False,
        # ---- continuous AR (MAR-style DiffLoss) ----
        diffloss_w: int = 1024,
        diffloss_d: int = 3,
        diff_steps: str = "100",
        diffloss_batch_mul: int = 4,
        diffloss_sample_clip_denoised: bool = True,
        continuous_head_type: str = 'diffloss',
        scale_loss_weighting: str = 'token',
        scale0_query_source: str = 'sos',
        scale0_start_source: str = 'transformer',
        stage3_context_mode: str = 'both',
        # ---- LR condition source: srvar_encoder / learned_lr_encoder / lr_vae ----
        lr_cond_source: str = 'srvar_encoder',
        learned_lr_encoder_width: int = 128,
    ):
        
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.Cvae * 4
        else:
            self.d_vae = vae_local.Cvae
        self.codebook_dim = self.d_vae
        self.V = vae_local.vocab_size
        self.bit_mask = None
        self.low_channel = low_channel # low-level-cond channel
        self.L = sum(pn ** 2 for pn in raw_scale_schedule)
        self.depth = depth 
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.learned_lr_encoder_width = int(learned_lr_encoder_width)
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        
        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        
        # # 好像没用
        # head_up_method = ''
        # word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        # if word_patch_size > 1:
        #     assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        # 暂时不用
        # self.customized_flash_attn = False
        
        customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        # The first autoregressive segment is not always a single token.  When
        # `patch_nums[0] == 4`, for example, scale[0] has 16 target tokens, so
        # the SOS conditioning token must be expanded to 16 positions to keep
        # x_BLC, the attention mask, and the DiffLoss target aligned.
        self.first_l = raw_scale_schedule[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.raw_scale_schedule):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
            
            
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}'
        # LR conditioning source. When 'srvar_encoder' we own the encoder/quant_conv
        # (initialised from the frozen HR VAE via init_LREncoder). When
        # 'learned_lr_encoder' we use a dedicated VAE-aligned encoder plus a trainable
        # detail branch. When 'lr_vae' the caller passes `low_f` from frozen LR_VAE.
        assert lr_cond_source in ('srvar_encoder', 'learned_lr_encoder', 'lr_vae'), \
            f"lr_cond_source must be 'srvar_encoder', 'learned_lr_encoder', or 'lr_vae', got {lr_cond_source!r}"
        self.lr_cond_source: str = lr_cond_source

        assert scale0_start_source in ('transformer', 'stage3'), \
            f"scale0_start_source must be 'transformer' or 'stage3', got {scale0_start_source!r}"
        assert stage3_context_mode in ('both', 'prefix_only'), \
            f"stage3_context_mode must be 'both' or 'prefix_only', got {stage3_context_mode!r}"
        self.scale0_start_source = scale0_start_source
        self.stage3_context_mode = stage3_context_mode
        self.stage3_uses_cross_attn = (
            self.scale0_start_source == 'stage3' and self.stage3_context_mode == 'both'
        )

        ddconfig = dict(
            dropout=vae_local.dropout, ch=vae_local.ch, z_channels=vae_local.Cvae,
            in_channels=getattr(vae_local, 'img_channels', 3), ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        self.learned_lr_encoder = None
        if self.lr_cond_source == 'srvar_encoder' and not self.stage3_uses_cross_attn:
            self.encoder = Encoder(double_z=False, **ddconfig)
            self.quant_conv = torch.nn.Conv2d(
                vae_local.Cvae, vae_local.Cvae,
                vae_local.quant_conv_ks, stride=1, padding=vae_local.quant_conv_ks // 2,
            )
        elif self.lr_cond_source == 'learned_lr_encoder' and not self.stage3_uses_cross_attn:
            self.encoder = None
            self.quant_conv = None
            self.learned_lr_encoder = LearnedLREncoder(
                ddconfig=ddconfig,
                quant_conv_ks=vae_local.quant_conv_ks,
                detail_width=self.learned_lr_encoder_width,
            )
        else:
            # Skip building the local encoder; LR_VAE is fed in externally.
            self.encoder = None
            self.quant_conv = None

        self.use_ref = use_ref
        if use_ref:
            assert self.lr_cond_source == 'srvar_encoder', \
                'use_ref is only supported with lr_cond_source=srvar_encoder.'
            assert not self.stage3_uses_cross_attn, \
                'use_ref is not used when stage3_context_mode=both replaces cross-attn KV.'
            self.encoder_ref = Encoder(double_z=False, **ddconfig)
            self.quant_conv_ref = torch.nn.Conv2d(
                vae_local.Cvae, vae_local.Cvae,
                vae_local.quant_conv_ks, stride=1, padding=vae_local.quant_conv_ks // 2,
            )

        # Held for backward compat with existing checkpoints / metric.py wiring;
        # new training path always uses the DiffLoss head below regardless.
        self.use_diff = True
        self.diffloss_batch_mul = int(max(1, diffloss_batch_mul))
        assert continuous_head_type in ('diffloss', 'mse'), \
            f"continuous_head_type must be 'diffloss' or 'mse', got {continuous_head_type!r}"
        self.continuous_head_type = continuous_head_type
        assert scale_loss_weighting in ('token', 'equal_scale'), \
            f"scale_loss_weighting must be 'token' or 'equal_scale', got {scale_loss_weighting!r}"
        assert scale0_query_source in ('sos', 'low_f_pool'), \
            f"scale0_query_source must be 'sos' or 'low_f_pool', got {scale0_query_source!r}"
        self.scale_loss_weighting = scale_loss_weighting
        self.scale0_query_source = scale0_query_source
        self.latest_per_scale_stats: List[Dict[str, float]] = []

        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.low_len = low_len
        
        # [inp & position embedding]
        
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        
        self.D = self.C
        
        cfg_uncond = torch.empty(self.low_len, self.low_channel)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.low_channel ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)
        
        # TODO:Neesky 是否考虑修改，因为这里不是文本了，而是图片的token。可能不适用于这种RMS归一化和Pool
        self.low_norm = FastRMSNorm(self.low_channel, elementwise_affine=True, eps=norm_eps)
        self.low_proj_for_sos = TextAttentivePool(self.low_channel, self.D)
        self.low_proj_for_ca = nn.Sequential(
            nn.Linear(self.low_channel, self.D),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.D, self.D),
        )
        # Only used when transformer predicts scale[0] with low_f_pool queries.
        # stage3 start uses _build_stage3_scale0_prefix; sos uses global SOS only.
        self.low_proj_for_scale0 = None
        if self.scale0_start_source == 'transformer' and self.scale0_query_source == 'low_f_pool':
            self.low_proj_for_scale0 = nn.Linear(self.low_channel, self.D)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C)) #SOS pos embeding
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        # TODO:Neesky 是否考虑修改，不知道啥用
        if self.rope2d_each_sa_layer:
            # SRtrainer may fall back to [(1,pn,pn)] when patch_nums != dynamic_resolution template length.
            fallback_scale_schedule = tuple((1, pn, pn) for pn in raw_scale_schedule)
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(
                dim=self.C // self.num_heads,
                dynamic_resolution_h_w=dynamic_resolution_h_w,
                pad_to_multiplier=self.pad_to_multiplier,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                extra_scale_schedules=[fallback_scale_schedule],
            )
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        
        # TODO:Neesky len(patch_nums)可能也是raw_scale_schedule，也就是10
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # pos_1LC = []
        # for i, pn in enumerate(self.raw_scale_schedule):
        #     pe = torch.empty(1, pn*pn, self.C)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        # pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        # assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        # self.pos_1LC = nn.Parameter(pos_1LC)
        # # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        # self.lvl_embed = nn.Embedding(len(self.raw_scale_schedule), self.C)
        # nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
            
         # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = CrossAttnBlock(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)
        
        # No discrete logits head: var now predicts continuous per-token vectors via
        # the DiffLoss head below. We still construct a tiny AdaLN-before-head module
        # to project the backbone output into the `z` space used as DiffLoss condition,
        # because it preserves the existing CFG / AdaLN structure of the codebase.
        self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
        # Linear C -> C: gives DiffLoss a dedicated projection without bloating params.
        self.head = nn.Linear(self.C, self.C)
        if self.continuous_head_type == 'mse':
            self.direct_head = nn.Linear(self.C, vae_local.Cvae)
        
        self.num_block_chunks = int(block_chunks or 1)
        assert self.num_block_chunks >= 1, f'block_chunks must be >= 1, got {block_chunks}'
        self.num_blocks_in_a_chunk = depth // self.num_block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, block_chunks={self.num_block_chunks}")
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        if self.continuous_head_type == 'diffloss':
            # MAR-style per-token DiffLoss head.
            # target_channels = Cvae   (continuous latent dim per token)
            # z_channels      = self.C (DiffLoss condition dim, = embed_dim)
            self.diffloss = DiffLoss(
                target_channels=vae_local.Cvae,
                z_channels=self.C,
                depth=int(diffloss_d),
                width=int(diffloss_w),
                num_sampling_steps=str(diff_steps),
                grad_checkpointing=(self.checkpointing == 'full-block'),
                sample_clip_denoised=bool(diffloss_sample_clip_denoised),
            )
        print(
            f'[srvar config] continuous_head_type={self.continuous_head_type}, '
            f'scale0_query_source={self.scale0_query_source}, '
            f'scale0_start_source={self.scale0_start_source}, '
            f'low_proj_for_scale0={"built" if self.low_proj_for_scale0 is not None else "skipped"}, '
            f'stage3_context_mode={self.stage3_context_mode}, '
            f'lr_cond_source={self.lr_cond_source}, '
            f'learned_lr_encoder_width={self.learned_lr_encoder_width}, '
            f'scale_loss_weighting={self.scale_loss_weighting}, '
            f'diffloss_batch_mul={self.diffloss_batch_mul}, '
            f'diffloss_sample_clip_denoised={bool(diffloss_sample_clip_denoised)}',
            flush=True,
        )
    
    def compile_flex_attn(self):
        
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "infinity_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                
                print("patchs_nums_tuple:")
                print(patchs_nums_tuple, mask_type, self.batch_size, self.num_heads, aligned_L, auto_padding)
                
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """Project transformer hidden state into the DiffLoss `z` condition space.

        The name `get_logits` is kept for parity with the legacy codebase, but in the
        continuous-AR head this no longer returns logits over a vocabulary; instead
        it returns the per-token condition `z` of dimension `self.C`.
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def _encode_lr_to_low_f(
        self,
        inp_B3HW_low: torch.Tensor,
        ref_B3HW: Optional[torch.Tensor],
        low_f_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce the `low_f` token sequence `[B, low_len, Cvae]` from LR input.

        - With `lr_cond_source='srvar_encoder'`, run SRVAR's local encoder/quant_conv.
        - With `lr_cond_source='learned_lr_encoder'`, run the dedicated LR encoder
          with a trainable detail residual branch.
        - With `lr_cond_source='lr_vae'`, the caller must pass `low_f_override`
          (already `[B, low_len, Cvae]`, e.g. flattened `lr_vae.encode_to_posterior_mean(LR)`).
        Optionally concat a reference image's tokens (only with srvar_encoder).
        """
        if self.lr_cond_source == 'lr_vae':
            assert low_f_override is not None, \
                'lr_cond_source=lr_vae requires the caller to pass `low_f_override`.'
            assert low_f_override.dim() == 3, \
                f'expected low_f_override [B,L,C], got {low_f_override.shape}'
            return low_f_override

        if self.lr_cond_source == 'learned_lr_encoder':
            assert self.learned_lr_encoder is not None
            B = inp_B3HW_low.shape[0]
            low_f = self.learned_lr_encoder(inp_B3HW_low)
            low_f = low_f.permute(0, 2, 3, 1).reshape(B, -1, low_f.shape[1])
            return low_f

        assert self.encoder is not None and self.quant_conv is not None
        B = inp_B3HW_low.shape[0]
        low_f = self.quant_conv(self.encoder(inp_B3HW_low))     # [B, C, h, w]
        low_f = low_f.permute(0, 2, 3, 1).reshape(B, -1, low_f.shape[1])
        if self.use_ref:
            assert ref_B3HW is not None, 'use_ref=True but ref_B3HW=None.'
            ref_f = self.quant_conv_ref(self.encoder_ref(ref_B3HW))
            ref_f = ref_f.permute(0, 2, 3, 1).reshape(B, -1, ref_f.shape[1])
            low_f = torch.cat((low_f, ref_f), dim=1)
        return low_f

    def _flatten_stage3_s0(
        self,
        stage3_s0: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Flatten frozen stage3 `s0_pred` to `[B, first_l, Cvae]` tokens."""
        assert stage3_s0 is not None, 'scale0_start_source=stage3 requires `stage3_s0`.'
        assert stage3_s0.dim() == 4, f'expected stage3_s0 [B,C,h,w], got {stage3_s0.shape}'
        B, C, H, W = stage3_s0.shape
        pn_t, pn_h, pn_w = scale_schedule[0]
        assert pn_t == 1, f'stage3 scale0 expects image scale[0], got {scale_schedule[0]}'
        assert (H, W) == (pn_h, pn_w), (
            f'stage3_s0 spatial {H}x{W} != scale[0] {pn_h}x{pn_w}; '
            f'check --stage3_latent_size and --patch_nums.'
        )
        assert C == self.d_vae, f'stage3_s0 channels {C} != VAE token dim {self.d_vae}'
        first_l = int(pn_t * pn_h * pn_w)
        assert first_l == self.first_l, (
            f'first scale token count {first_l} != model first_l={self.first_l}; '
            f'check raw_scale_schedule={self.raw_scale_schedule} vs scale_schedule={scale_schedule}'
        )
        return stage3_s0.reshape(B, C, -1).transpose(1, 2).contiguous()

    def _select_condition_low_f(
        self,
        inp_B3HW_low: Optional[torch.Tensor],
        ref_B3HW: Optional[torch.Tensor],
        low_f_override: Optional[torch.Tensor],
        stage3_s0: Optional[torch.Tensor],
        scale_schedule: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Choose cross-attn KV tokens for the active scale0-start mode."""
        if self.stage3_uses_cross_attn:
            return self._flatten_stage3_s0(stage3_s0, scale_schedule)
        return self._encode_lr_to_low_f(inp_B3HW_low, ref_B3HW, low_f_override)

    def _build_scale0_queries(
        self,
        sos: torch.Tensor,
        low_f_BLC: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Build the initial query tokens for scale[0].

        The legacy `sos` path keeps exact old behavior. `low_f_pool` injects a
        spatially-pooled LR latent grid so the first 4x4 tokens do not all start
        from the same global vector.
        """
        B = sos.shape[0]
        pn_t, pn_h, pn_w = scale_schedule[0]
        first_l = int(pn_t * pn_h * pn_w)
        assert first_l == self.first_l, (
            f'first scale token count {first_l} != model first_l={self.first_l}; '
            f'check raw_scale_schedule={self.raw_scale_schedule} vs scale_schedule={scale_schedule}'
        )

        global_sos = sos.unsqueeze(1).expand(B, first_l, -1)
        pos = self.pos_start.expand(B, first_l, -1)
        if self.scale0_query_source == 'sos':
            return global_sos + pos

        assert pn_t == 1, f'low_f_pool scale0 query expects 2D scale[0], got {scale_schedule[0]}'
        source_BLC = low_f_BLC
        if self.use_ref and source_BLC.shape[1] % 2 == 0:
            source_BLC = source_BLC[:, :source_BLC.shape[1] // 2]

        low_len = source_BLC.shape[1]
        low_side = math.isqrt(low_len)
        if low_side * low_side != low_len:
            raise ValueError(
                f'scale0_query_source=low_f_pool requires a square LR token grid; '
                f'got low_len={low_len}, full_low_len={low_f_BLC.shape[1]}, use_ref={self.use_ref}'
            )

        source_BChw = source_BLC.transpose(1, 2).reshape(B, source_BLC.shape[-1], low_side, low_side)
        pooled = F.adaptive_avg_pool2d(source_BChw, output_size=(pn_h, pn_w))
        pooled_BLC = pooled.flatten(2).transpose(1, 2).contiguous()
        assert self.low_proj_for_scale0 is not None, (
            'low_proj_for_scale0 is only built for '
            "scale0_start_source='transformer' and scale0_query_source='low_f_pool'."
        )
        return global_sos + self.low_proj_for_scale0(pooled_BLC) + pos

    def _build_stage3_scale0_prefix(
        self,
        stage3_s0: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Project frozen stage3 `s0_pred` as observed scale[0] self-attn prefix."""
        stage3_BLC = self._flatten_stage3_s0(stage3_s0, scale_schedule)
        B, first_l, _ = stage3_BLC.shape
        assert first_l == self.pos_start.shape[1], (
            f'stage3 prefix length {first_l} != pos_start length {self.pos_start.shape[1]}'
        )
        return self.word_embed(self.norm0_ve(stage3_BLC)) + self.pos_start.expand(B, first_l, -1)
    
    def get_scale_logits(self,
                   si,
                   last_stage,
                   cond_BD_or_gss,
                   ca_kv,
                   scale_schedule,
                   B,
                   need_to_pad,
                   attn_fn,
                   cache_now
                   ):

        for b in self.unregistered_blocks: 
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching_now(cache_now)

        for block_idx, b in enumerate(self.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if self.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
            if not self.add_lvl_embeding_only_first_block: 
                last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)


        return  last_stage[:B]

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        inp_B3HW_low: Optional[torch.Tensor] = None,
        ref_B3HW: Optional[torch.Tensor] = None,
        low_f_override: Optional[torch.Tensor] = None,
        stage3_s0: Optional[torch.Tensor] = None,
        B: int = 1,
        g_seed: Optional[int] = None,
        ret_img: bool = False,
        trunk_scale: int = 1000,
        temperature: float = 1.0,
        cfg: float = 1.0,
        return_fhat: bool = False,
    ):
        """Multi-scale autoregressive inference with continuous DiffLoss head.

        For each scale `si`, the transformer produces a per-token condition `z`,
        we call `self.diffloss.sample(z, temperature, cfg)` to sample the
        continuous token, accumulate via `vae.quantize.get_next_autoregressive_input`,
        and feed the next-scale teacher-forcing input back into the transformer.

        Returns one of:
            (ret_list, []) if not `ret_img` and not `return_fhat`
            (ret_list, []) where the second slot is `(accu_BChw,)` if `return_fhat`
            (ret_list, [], img) if `ret_img`; img is `[B, H, W, img_channels]` uint8.
        """
        if g_seed is not None:
            self.rng.manual_seed(g_seed)

        assert scale_schedule is not None and vae is not None

        # ---- 1. LR conditioning ----
        device = (
            inp_B3HW_low.device if inp_B3HW_low is not None
            else low_f_override.device if low_f_override is not None
            else stage3_s0.device
        )
        low_f = self._select_condition_low_f(
            inp_B3HW_low, ref_B3HW, low_f_override, stage3_s0, scale_schedule
        )
        lowLen = low_f.shape[1]
        lens = torch.full((B,), lowLen, dtype=torch.int32, device=device)
        max_seqlen_k = lens.max()
        cu_seqlens_k = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.int32, device=device), lens]), dim=0
        ).to(dtype=torch.int32)

        kv_compact = low_f.reshape(-1, low_f.shape[-1])
        kv_compact = self.low_norm(kv_compact)
        low_f_BLC = kv_compact.reshape(B, lowLen, -1)
        sos = cond_BD = self.low_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
        kv_compact = self.low_proj_for_ca(kv_compact)
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()

        # ---- 2. AR loop ----
        first_l = int(scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2])
        assert first_l == self.first_l, (
            f'first scale token count {first_l} != model first_l={self.first_l}; '
            f'check raw_scale_schedule={self.raw_scale_schedule} vs scale_schedule={scale_schedule}'
        )
        stage3_mode = self.scale0_start_source == 'stage3'
        if stage3_mode:
            last_stage = self._build_stage3_scale0_prefix(stage3_s0, scale_schedule)
        else:
            last_stage = self._build_scale0_queries(sos, low_f_BLC, scale_schedule)
        accu_BChw = sos.new_zeros(B, vae.Cvae, self.raw_scale_schedule[-1], self.raw_scale_schedule[-1])
        num_stages_minus_1 = len(scale_schedule) - 1
        need_to_pad = 0
        attn_fn = None

        ret: List[torch.Tensor] = []
        for b in self.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

        try:
            start_si = 0
            if stage3_mode:
                s0 = stage3_s0.contiguous()
                ret.append(s0)
                if trunk_scale <= 1 or num_stages_minus_1 == 0:
                    accu_BChw, _ = vae.quantize.get_next_autoregressive_input(
                        0, len(self.raw_scale_schedule), accu_BChw, s0,
                    )
                else:
                    # Warm self-attn KV cache with the observed stage3 s0 prefix,
                    # then ignore transformer output for scale[0].
                    _ = self.get_scale_logits(
                        si=0, last_stage=last_stage, cond_BD_or_gss=cond_BD_or_gss,
                        ca_kv=ca_kv, scale_schedule=scale_schedule,
                        B=B, need_to_pad=need_to_pad, attn_fn=attn_fn, cache_now=True,
                    )
                    accu_BChw, last_stage_fhat = vae.quantize.get_next_autoregressive_input(
                        0, len(self.raw_scale_schedule), accu_BChw, s0,
                    )
                    last_stage = last_stage_fhat.view(B, vae.Cvae, -1).transpose(1, 2)
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
                start_si = 1

            for si in range(start_si, len(scale_schedule)):
                pn = scale_schedule[si]
                if si >= trunk_scale:
                    break
                pn_t, pn_h, pn_w = pn
                num_pn = int(pn_t * pn_h * pn_w)

                # Transformer hidden state for this scale's tokens.
                BlV = self.get_scale_logits(
                    si=si, last_stage=last_stage, cond_BD_or_gss=cond_BD_or_gss,
                    ca_kv=ca_kv, scale_schedule=scale_schedule,
                    B=B, need_to_pad=need_to_pad, attn_fn=attn_fn, cache_now=True,
                )                                                       # [B, pn_tokens, D]
                z_BlD = self.get_logits(BlV[:B], cond_BD[:B])           # [B, pn_tokens, D]

                # Per-token continuous latent prediction.
                z_flat = z_BlD.reshape(-1, z_BlD.shape[-1]).contiguous() # [B*pn, D]
                if self.continuous_head_type == 'mse':
                    h_flat = self.direct_head(z_flat)
                else:
                    h_flat = self.diffloss.sample(z_flat, temperature=temperature, cfg=cfg)
                h_BChw = h_flat.reshape(B, pn_h, pn_w, vae.Cvae).permute(0, 3, 1, 2).contiguous()

                ret.append(h_BChw)

                # Accumulate and prepare next-scale teacher-forcing input.
                accu_BChw, last_stage_fhat = vae.quantize.get_next_autoregressive_input(
                    si, len(self.raw_scale_schedule), accu_BChw, h_BChw,
                )
                if si != num_stages_minus_1:
                    last_stage = last_stage_fhat.view(B, vae.Cvae, -1).transpose(1, 2)
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
        finally:
            for b in self.unregistered_blocks:
                (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

        if return_fhat:
            return ret, (accu_BChw,), None

        if not ret_img:
            return ret, [], None

        img = vae.fhat_to_img(accu_BChw)
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
        return ret, [], img

    
    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC
    
    def forward(
        self,
        inp_B3HW_low: torch.Tensor,
        ms_h_target: List[torch.Tensor],
        ms_x_input: Optional[torch.Tensor],
        scale_schedule: List[Tuple[int, int, int]],
        ref_B3HW: Optional[torch.Tensor] = None,
        low_f_override: Optional[torch.Tensor] = None,
        stage3_s0: Optional[torch.Tensor] = None,
        cfg_infer: bool = False,
        scale0_loss_mask: bool = False,
    ) -> torch.Tensor:
        """Training forward.

        Args:
            inp_B3HW_low:   `[B, img_channels, H, W]` LR image, used only when
                            `lr_cond_source='srvar_encoder'`.
            ms_h_target:    list of `[B, C, pn, pn]` per-scale posterior means; the
                            DiffLoss targets.
            ms_x_input:     `[B, L-first_l, C]` teacher-forcing continuous tokens
                            (already comes from VAE's `f_to_var_input_continuous`).
                            May be `None` if `len(scale_schedule)==1`.
            scale_schedule: list of (t, h, w) per scale.
            ref_B3HW:       optional reference image (use_ref=True only).
            low_f_override: `[B, low_len, C]` raw LR tokens; required when
                            `lr_cond_source='lr_vae'`, ignored otherwise.
            stage3_s0:      frozen stage3 LR->scale[0] latent used when
                            `scale0_start_source='stage3'`.
            cfg_infer:      if True, disable training-time CFG dropout (used by infer).
            scale0_loss_mask: if True, drop scale[0] tokens from the DiffLoss target/z.

        Returns:
            scalar DiffLoss.
        """
        # ---- sanity ----
        SN = len(scale_schedule)
        if self.scale0_start_source == 'stage3':
            scale0_loss_mask = True
        assert len(ms_h_target) == SN, \
            f'len(ms_h_target)={len(ms_h_target)} != len(scale_schedule)={SN}'
        if ms_x_input is not None:
            B = ms_x_input.shape[0]
            ms_x_input = ms_x_input.float()
        else:
            B = ms_h_target[0].shape[0]
        device = ms_h_target[0].device

        # ---- 1. LR conditioning ----
        with torch.amp.autocast('cuda', enabled=False):
            low_f = self._select_condition_low_f(
                inp_B3HW_low, ref_B3HW, low_f_override, stage3_s0, scale_schedule
            )
            lowLen = low_f.shape[1]
            assert lowLen <= self.cfg_uncond.shape[0], \
                f'low_len={lowLen} exceeds cfg_uncond buffer length={self.cfg_uncond.shape[0]}; ' \
                f'increase --tlen.'
            lens = torch.full((B,), lowLen, dtype=torch.int32, device=device)
            max_seqlen_k = lens.max()
            cu_seqlens_k = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int32, device=device), lens]), dim=0
            ).to(dtype=torch.int32)

            kv_compact = low_f.reshape(-1, low_f.shape[-1])      # [B*low_len, C]
            # Classifier-free guidance dropout (training only).
            total = 0
            if not cfg_infer:
                for le in lens:
                    if random.random() < self.cond_drop_rate:
                        kv_compact[total:total + le] = self.cfg_uncond[:le]
                    total += int(le.item())
            must_on_graph = self.cfg_uncond[0, 0] * 0           # keep cfg_uncond in graph
            kv_compact = self.low_norm(kv_compact).contiguous()
            low_f_BLC = kv_compact.reshape(B, lowLen, -1)
            sos = cond_BD = self.low_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
            kv_compact = self.low_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()

            # ---- 2. build x_BLC = [SOS_for_scale0, word_embed(ms_x_input)] ----
            target_l = sum(int(pn[0] * pn[1] * pn[2]) for pn in scale_schedule)
            first_l = int(scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2])
            assert first_l == self.first_l, (
                f'first scale token count {first_l} != model first_l={self.first_l}; '
                f'check raw_scale_schedule={self.raw_scale_schedule} vs scale_schedule={scale_schedule}'
            )
            expected_ms_x_l = target_l - first_l
            if ms_x_input is not None:
                assert ms_x_input.shape[1] == expected_ms_x_l, (
                    f'ms_x_input length {ms_x_input.shape[1]} != expected L-first_l '
                    f'{expected_ms_x_l}; target_l={target_l}, first_l={first_l}, '
                    f'scale_schedule={scale_schedule}'
                )
            else:
                assert expected_ms_x_l == 0, (
                    f'ms_x_input=None but scale_schedule has {expected_ms_x_l} '
                    f'teacher-forcing tokens after scale[0].'
                )
            if self.scale0_start_source == 'stage3':
                sos = self._build_stage3_scale0_prefix(stage3_s0, scale_schedule)
            else:
                sos = self._build_scale0_queries(sos, low_f_BLC, scale_schedule)
            if ms_x_input is not None:
                x_BLC = torch.cat(
                    (sos, self.word_embed(self.norm0_ve(ms_x_input))), dim=1
                )
            else:
                x_BLC = sos

            l_end = x_BLC.shape[1]
            assert l_end == target_l, (
                f'x_BLC length {l_end} != target token count {target_l}; '
                f'first_l={first_l}, ms_x_input={None if ms_x_input is None else tuple(ms_x_input.shape)}'
            )
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end

            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([
                    torch.full((pn[0] * pn[1] * pn[2],), i) for i, pn in enumerate(scale_schedule)
                ]).view(1, l_end, 1)
                dT = d.transpose(1, 2)
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)

        attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)] if self.use_flex_attn else None

        # ---- 3. transformer blocks ----
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if (self.add_lvl_embeding_only_first_block and i == 0) or not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(
                        b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn,
                        scale_schedule, self.rope2d_freqs_grid, use_reentrant=False,
                    )
                else:
                    x_BLC = b(
                        x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_bias_or_two_vector,
                        attn_fn=attn_fn, scale_schedule=scale_schedule,
                        rope2d_freqs_grid=self.rope2d_freqs_grid,
                    )
        else:
            for i, chunk in enumerate(self.block_chunks):
                if (self.add_lvl_embeding_only_first_block and i == 0) or not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(
                    x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv,
                    attn_bias_or_two_vector=attn_bias_or_two_vector,
                    attn_fn=attn_fn, scale_schedule=scale_schedule,
                    checkpointing_full_block=checkpointing_full_block,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                )

        # ---- 4. project to DiffLoss condition `z` ----
        z_BLD = self.get_logits(x_BLC[:, :l_end], cond_BD)        # [B, L, D=C]

        # ---- 5. build target tensor and call the continuous head loss ----
        # Target tokens are laid out scale-by-scale to match the transformer order.
        # For scale `si`, target[si] has `pn^2` tokens of dim Cvae.
        target_list: List[torch.Tensor] = []
        for si, pn in enumerate(scale_schedule):
            h = ms_h_target[si]                                    # [B, Cvae, pn_h, pn_w]
            target_list.append(
                h.reshape(h.shape[0], h.shape[1], -1).transpose(1, 2).contiguous()
            )                                                      # [B, pn^2, Cvae]
        target_BLC = torch.cat(target_list, dim=1)                 # [B, L, Cvae]
        assert target_BLC.shape[1] == z_BLD.shape[1], \
            f'target tokens {target_BLC.shape} != z tokens {z_BLD.shape}'

        # Flatten over B*L for DiffLoss.
        z_flat = z_BLD.reshape(-1, z_BLD.shape[-1]).contiguous()
        tgt_flat = target_BLC.reshape(-1, target_BLC.shape[-1]).contiguous()

        # Optional mask/weight: Plan-A can drop scale[0], and equal_scale makes
        # each scale contribute an equal average regardless of token count.
        loss_mask: Optional[torch.Tensor] = None
        if scale0_loss_mask or self.scale_loss_weighting == 'equal_scale':
            mask_BL = torch.ones(B, target_BLC.shape[1], device=device, dtype=torch.float32)
            ptr = 0
            for si, pn in enumerate(scale_schedule):
                n = int(pn[0] * pn[1] * pn[2])
                if scale0_loss_mask and si == 0:
                    mask_BL[:, ptr:ptr + n] = 0.0
                elif self.scale_loss_weighting == 'equal_scale':
                    mask_BL[:, ptr:ptr + n] = 1.0 / max(1, n)
                ptr += n
            loss_mask = mask_BL.reshape(-1)

        self.latest_per_scale_stats = []
        if self.continuous_head_type == 'mse':
            pred_flat = self.direct_head(z_flat)
            loss_per_token = (pred_flat.float() - tgt_flat.float()).square().mean(dim=1)
            self._record_per_scale_mse_stats(
                loss_per_token.reshape(B, target_BLC.shape[1]),
                scale_schedule,
            )
            if loss_mask is not None:
                return (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            return loss_per_token.mean()

        # MAR-style diffloss_batch_mul: repeat the per-token pairs to reduce variance.
        mul = self.diffloss_batch_mul if self.training else 1
        if mul > 1:
            z_flat = z_flat.repeat_interleave(mul, dim=0)
            tgt_flat = tgt_flat.repeat_interleave(mul, dim=0)
            if loss_mask is not None:
                loss_mask = loss_mask.repeat_interleave(mul, dim=0)

        loss = self.diffloss(target=tgt_flat, z=z_flat, mask=loss_mask)
        return loss

    def _record_per_scale_mse_stats(
        self,
        loss_per_token_BL: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
    ) -> None:
        """Cache raw per-scale latent MSE diagnostics for the latest forward pass."""
        stats: List[Dict[str, float]] = []
        with torch.no_grad():
            ptr = 0
            for si, pn in enumerate(scale_schedule):
                n = int(pn[0] * pn[1] * pn[2])
                mse = loss_per_token_BL[:, ptr:ptr + n].detach().float().mean()
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                stats.append({
                    'scale': float(si),
                    'tokens': float(n),
                    'h': float(pn[1]),
                    'w': float(pn[2]),
                    'mse': float(mse.item()),
                    'psnr': float(psnr.item()),
                })
                ptr += n
        self.latest_per_scale_stats = stats
        
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]

        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm (AdaLN that produces the DiffLoss condition projection)
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        # init head's projection
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                if self.head.bias is not None:
                    self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                if self.head[-1].bias is not None:
                    self.head[-1].bias.data.zero_()
            if hasattr(self, 'direct_head'):
                self.direct_head.weight.data.mul_(scale_head)
                if self.direct_head.bias is not None:
                    self.direct_head.bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks): 
            sab: Union[SelfAttnBlock, CrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:

                sab.sa.proj.weight.data.mul_(scale)
                sab.ca.proj.weight.data.mul_(scale)
                
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
        if hasattr(self, 'diffloss'):
            self._assert_diffloss_final_layer_zero()

    def _assert_diffloss_final_layer_zero(self):
        final_layer = self.diffloss.net.final_layer
        linear = final_layer.linear
        ada = final_layer.adaLN_modulation[-1]
        with torch.no_grad():
            linear_weight_norm = float(linear.weight.detach().norm().item())
            linear_bias_norm = float(linear.bias.detach().norm().item()) if linear.bias is not None else 0.0
            ada_weight_norm = float(ada.weight.detach().norm().item())
            ada_bias_norm = float(ada.bias.detach().norm().item()) if ada.bias is not None else 0.0
        print(
            '[diffloss init] '
            f'final_linear_w={linear_weight_norm:.6e} final_linear_b={linear_bias_norm:.6e} '
            f'final_adaln_w={ada_weight_norm:.6e} final_adaln_b={ada_bias_norm:.6e}',
            flush=True,
        )
        eps = 1e-8
        assert linear_weight_norm <= eps and linear_bias_norm <= eps and ada_weight_norm <= eps and ada_bias_norm <= eps, (
            'DiffLoss final layer lost its MAR zero initialization; '
            f'linear_w={linear_weight_norm}, linear_b={linear_bias_norm}, '
            f'ada_w={ada_weight_norm}, ada_b={ada_bias_norm}'
        )
    
    def init_weights(self, conv_std_or_gain: float = 0.02, other_std: float = 0.02):
        """
        :param model: the model to be inited
        :param conv_std_or_gain: how to init every conv layer `m`
            > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            < 0: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
        :param other_std: how to init every linear layer or embedding layer
            use nn.init.trunc_normal_(m.weight.data, std=other_std)
        """
        skip = abs(conv_std_or_gain) > 10
        if skip: return
        print(f'[init_weights] {type(self).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
        diffloss_modules = set(self.diffloss.modules()) if hasattr(self, 'diffloss') else set()
        for m in self.modules():
            if m in diffloss_modules:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=other_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=other_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain) if conv_std_or_gain > 0 else nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)   # todo: StyleSwin: (..., gain=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)
                if m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.)
    
    def init_LREncoder(self, vae_local: VQVAE):
        """Copy encoder/quant_conv weights from the frozen HR VAE into SRVAR's own
        encoder. `learned_lr_encoder` also copies the VAE-aligned trunk, while its
        residual detail branch remains zero-initialized and fully trainable.
        """
        if self.stage3_uses_cross_attn:
            print("[init_LREncoder] skipped (stage3_context_mode=both uses stage3_s0 as cross-attn KV).")
            return
        if self.lr_cond_source == 'learned_lr_encoder':
            assert self.learned_lr_encoder is not None
            self.learned_lr_encoder.init_from_vae(vae_local)
            print("[init_LREncoder] learned_lr_encoder trunk initialized from frozen VAE; detail branch remains trainable.")
            return
        if self.lr_cond_source != 'srvar_encoder':
            print(f"[init_LREncoder] skipped (lr_cond_source={self.lr_cond_source}).")
            return
        assert self.encoder is not None and self.quant_conv is not None
        self.encoder.load_state_dict(vae_local.encoder.state_dict())
        self.quant_conv.load_state_dict(vae_local.quant_conv.state_dict())

        if self.use_ref:
            self.encoder_ref.load_state_dict(vae_local.encoder.state_dict())
            self.quant_conv_ref.load_state_dict(vae_local.quant_conv.state_dict())

        print("LRencoder loaded from vae_local")
        
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

if __name__ == "__main__":
    # Smoke test removed: relied on the legacy discrete VAR API.
    # See metric.py / SRtrainer.train_step for the new entry points.
    pass
