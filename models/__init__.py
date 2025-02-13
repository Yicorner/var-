from typing import Tuple
import torch.nn as nn
from .var import VAR
from .quant import VectorQuantizer2
from .SRVAR import SRVAR
from .vqvae import VQVAE
from utils import arg_util
def build_vae_srvar(# Shared args
    args: arg_util.Args,
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    
) -> Tuple[VQVAE, SRVAR]:
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    
    gpt_kw = dict(
        low_channel=args.Ct5, low_len=args.tlen,
        norm_eps=args.norm_eps, rms_norm=args.rms,
        shared_aln=args.saln, head_aln=args.haln,
        cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
        cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau, cos_attn=args.cos, swiglu=args.swi,
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
        block_chunks = args.block_chunks,

    )
    if args.dp >= 0: gpt_kw['drop_path_rate'] = args.dp
    if args.hd > 0: gpt_kw['num_heads'] = args.hd
    
    print(f'[create gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    
    gpt_wo_ddp: SRVAR = SRVAR(**gpt_kw)
    # TODO：Neesky use_fsdp_model_ema 记得添加到args里
    if args.use_fsdp_model_ema:
        # gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
        raise NotImplementedError('ema not supported')
    else:
        gpt_wo_ddp_ema = None
        
    gpt_wo_ddp = gpt_wo_ddp.to(device)

    assert all(not p.requires_grad for p in vae_local.parameters())
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    
    return vae_local, gpt_wo_ddp, gpt_wo_ddp_ema
