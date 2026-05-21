from typing import Optional, Tuple

import torch
import torch.nn as nn

from .var import VAR
from .quant import ContinuousMultiScaleQuantizer, VectorQuantizer2
from .SRVAR import SRVAR
from .vqvae import VQVAE
from .lr_vae import LR_VAE
from utils import arg_util


def build_vae_srvar(
    args: arg_util.Args,
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    V: int = 0, Cvae: int = 32, ch: int = 128, share_quant_resi: int = 4,
) -> Tuple[VQVAE, SRVAR]:
    """Build the frozen continuous multi-scale VAE and the SRVAR backbone.

    The VAE is always constructed in `test_mode=True` and must be loaded from a
    myvaex stage2 checkpoint at the call site (`SRtrain.py`).
    """
    vae_local = VQVAE(
        vocab_size=V, z_channels=Cvae, ch=ch,
        test_mode=True, share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
        quant_resi=getattr(args, 'quant_resi', 0.5),
    ).to(device)

    srvar_kw = dict(
        low_channel=args.Ct5, low_len=args.tlen,
        norm_eps=args.norm_eps, rms_norm=args.rms,
        shared_aln=args.saln, head_aln=args.haln,
        cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
        cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau,
        cos_attn=args.cos, swiglu=args.swi,
        raw_scale_schedule=args.patch_nums,
        head_depth=args.dec,
        top_p=args.tp, top_k=args.tk,
        customized_flash_attn=args.flash, fused_mlp=args.fuse, fused_norm=args.fused_norm,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        batch_size=args.batch_size,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=None,
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
        block_chunks=args.block_chunks,
        use_ref=args.use_ref,
        # ---- continuous AR head ----
        diffloss_w=getattr(args, 'diffloss_w', 1024),
        diffloss_d=getattr(args, 'diffloss_d', 3),
        diff_steps=str(getattr(args, 'diff_steps', '100')),
        diffloss_batch_mul=getattr(args, 'diffloss_batch_mul', 4),
        diffloss_sample_clip_denoised=getattr(args, 'diffloss_sample_clip_denoised', True),
        continuous_head_type=getattr(args, 'continuous_head_type', 'diffloss'),
        scale_loss_weighting=getattr(args, 'scale_loss_weighting', 'token'),
        scale0_query_source=getattr(args, 'scale0_query_source', 'sos'),
        # ---- LR conditioning source ----
        lr_cond_source=getattr(args, 'lr_cond_source', 'srvar_encoder'),
    )
    if args.dp >= 0:
        srvar_kw['drop_path_rate'] = args.dp
    if args.hd > 0:
        srvar_kw['num_heads'] = args.hd

    print(f'[create srvar_wo_ddp] constructor kw={srvar_kw}\n')
    srvar_kw['vae_local'] = vae_local
    srvar_wo_ddp: SRVAR = SRVAR(**srvar_kw)
    srvar_wo_ddp = srvar_wo_ddp.to(device)

    assert all(not p.requires_grad for p in vae_local.parameters()), \
        'VAE must be fully frozen; check test_mode=True at construction.'
    assert all(p.requires_grad for _, p in srvar_wo_ddp.named_parameters()), \
        'SRVAR must have all parameters trainable.'

    return vae_local, srvar_wo_ddp


def build_lr_vae(
    args: arg_util.Args, device,
    Cvae: int = 32, ch: int = 128,
) -> LR_VAE:
    """Construct a frozen LR_VAE for the optional stage1 path.

    Only callers with non-empty `args.stage1_ckpt` should invoke this; checkpoint
    loading is performed in `SRtrain.py`.
    """
    lr_vae = LR_VAE(z_channels=Cvae, ch=ch, test_mode=True).to(device)
    assert all(not p.requires_grad for p in lr_vae.parameters()), \
        'LR_VAE must be fully frozen (test_mode=True).'
    return lr_vae
