import math
import random
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flex_attn import FlexAttn

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from models.basic import CrossAttnBlock,flash_attn_func, FastRMSNorm, SelfAttnBlock, flash_fused_op_installed, CrossAttention, precompute_rope2d_freqs_grid

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
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
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
        apply_spatial_patchify = 0,
        inference_mode=False,
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
        self.first_l = 1
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
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C)) #SOS pos embeding
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        # TODO:Neesky 是否考虑修改，不知道啥用
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
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
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
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
        
        # # 0. hyperparameters
        # assert embed_dim % num_heads == 0
        # self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        # self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        # self.cond_drop_rate = cond_drop_rate
        # self.prog_si = -1   # progressive training
        
        # self.patch_nums: Tuple[int] = patch_nums
        # self.L = sum(pn ** 2 for pn in self.patch_nums)
        # self.first_l = self.patch_nums[0] ** 2
        # self.begin_ends = []
        # cur = 0
        # for i, pn in enumerate(self.patch_nums):
        #     self.begin_ends.append((cur, cur+pn ** 2))
        #     cur += pn ** 2
        
        # self.num_stages_minus_1 = len(self.patch_nums) - 1
        # self.rng = torch.Generator(device=dist.get_device())
        
        # # 1. input (word) embedding
        # quant: VectorQuantizer2 = vae_local.quantize
        # self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        # self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        # self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # # 2. class embedding
        # init_std = math.sqrt(1 / self.C / 3)
        # self.num_classes = num_classes
        # self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        # self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        # nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        # self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        # nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # # 3. absolute position embedding
        # pos_1LC = []
        # for i, pn in enumerate(self.patch_nums):
        #     pe = torch.empty(1, pn*pn, self.C)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        # pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        # assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        # self.pos_1LC = nn.Parameter(pos_1LC)
        # # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        # self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        # nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # # 4. backbone blocks
        # self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        # self.drop_path_rate = drop_path_rate
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        # self.blocks = nn.ModuleList([
        #     AdaLNSelfAttn(
        #         cond_dim=self.D, shared_aln=shared_aln,
        #         block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
        #         attn_l2_norm=attn_l2_norm,
        #         flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        #     )
        #     for block_idx in range(depth)
        # ])
        
        # fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        # self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        # print(
        #     f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
        #     f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
        #     f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
        #     end='\n\n', flush=True
        # )
        
        # # 5. attention mask used in training (for masking out the future)
        # #    it won't be used in inference, since kv cache is enabled
        # d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        # dT = d.transpose(1, 2)    # dT: 11L
        # lvl_1L = dT[:, 0].contiguous()
        # self.register_buffer('lvl_1L', lvl_1L)
        # attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        # self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # # 6. classifier head
        # self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        # self.head = nn.Linear(self.C, self.V)
        
        
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
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))
    
    def get_scale_logits(self,
                   si,
                   last_stage,
                   cond_BD_or_gss,
                   ca_kv,
                   cond_BD,
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


        return  self.get_logits(last_stage[:B], cond_BD[:B])

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, 
        g_seed=None, 
        returns_vemb=0, 
        ret_img=False,              # 是否返回图片
        trunk_scale=1000,           # 控制图片最大的大小，大于这个不生成了
        beam_search_nums = 3,

        choose_min = "max_2max",
        score_compare = "max_2max",
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

        bs = B
            
        kv_compact = self.low_norm(kv_compact)
        sos = cond_BD = self.low_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        
        kv_compact = self.low_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list = []
        

        accu_BChw = sos.new_zeros(B, vae.Cvae, self.raw_scale_schedule[-1], self.raw_scale_schedule[-1])
        
        
        num_stages_minus_1 = len(scale_schedule)-1

        need_to_pad = 0
        attn_fn = None

        self.use_flex_attn = False
        if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

        for b in self.unregistered_blocks: 
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            if si >= trunk_scale:
                break
            num_pn = np.array(pn).prod()
            cur_L += num_pn
            nex_is = si+1

            logits_BlV = self.get_scale_logits(si=si, last_stage=last_stage, cond_BD_or_gss=cond_BD_or_gss, \
                                                        ca_kv = ca_kv, cond_BD=cond_BD, scale_schedule=scale_schedule, \
                                                        B=B, need_to_pad=need_to_pad, attn_fn=attn_fn, cache_now=True)
            
            # idx_Bl = logits_BlV.data.argmax(dim=-1)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=900, top_p=0.95, num_samples=1)[:, :, 0]
            
            beam_search_nums_modify= min(beam_search_nums,num_pn)
            if beam_search_nums > 0 and si != num_stages_minus_1:

                probs = F.softmax(logits_BlV, dim=-1)
                value_Bl, idx_Bl = probs.max(dim=-1)
                # print("idx_BL:",idx_Bl.shape, value_Bl.shape)
                if choose_min == "max":
                    min_value_value_Bls,min_idx_value_Bls = value_Bl.topk(beam_search_nums_modify, dim=-1, largest=False)
                elif choose_min == "max_2max":
                    top2_values, top2_indices = probs.topk(2, dim=-1)
                    prob_max_2max = 2 * top2_values[...,0] - top2_values[...,1]
                    min_value_value_Bls,min_idx_value_Bls = prob_max_2max.topk(beam_search_nums_modify, dim=-1, largest=False)
                
                # print("min_Bls:",min_value_value_Bls.shape, min_idx_value_Bls.shape)
                # print("min_value_value_Bls:",min_value_value_Bls)
                # print("min_idx_value_Bls:",min_idx_value_Bls)

                # max_value_value_Bls,max_idx_value_Bls = value_Bl.topk(beam_search_nums_modify, dim=-1, largest=True)
                # print("max_Bls:",max_value_value_Bls.shape, max_idx_value_Bls.shape)
                # print("max_value_value_Bls:",max_value_value_Bls)
                # print("max_idx_value_Bls:",max_idx_value_Bls)

                
                beam_find_best_idx_Bl = idx_Bl.clone()
                beam_find_best_score_Bl = torch.zeros(B,device=accu_BChw.device,dtype=accu_BChw.dtype)
                for beam_search_idx in  range(2 ** beam_search_nums_modify):
                    beam_idx_Bl = idx_Bl.clone()
                    beam_accu_BChw = accu_BChw.clone()
                    beam_choose_p = torch.ones((B,1),device=accu_BChw.device,dtype=accu_BChw.dtype)
                    for _search_idx in range(beam_search_nums_modify):

                        pos_idx = min_idx_value_Bls[...,_search_idx].unsqueeze(-1)
                        # print("pos_idx:",pos_idx,"min_idx_value_Bls",min_idx_value_Bls.shape)
                        # print("probs:",probs.shape)
                        batch_indices = torch.arange(probs.shape[0], device=probs.device).view(-1, 1).expand(-1, pos_idx.shape[1])
                        top2_values, top2_indices = probs[batch_indices,pos_idx].topk(2, dim=-1)
                        # print("top2_indices:",top2_indices)
                        # 
                        if (beam_search_idx & 2**_search_idx) == 0:
                            beam_choose_p = beam_choose_p + top2_values[...,0]
                            continue 
                                               
                        beam_choose_p = beam_choose_p + top2_values[...,1]
                        beam_idx_Bl[batch_indices,pos_idx] = top2_indices[...,1]
                        # print(f"pos_idx:{pos_idx} chooose top2_indices:{top2_indices}")

                    h_BChw = vae.quantize.embedding(beam_idx_Bl).float()   # BlC
                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][1], scale_schedule[si][2])
                    beam_accu_BChw, beam_last_stage = vae.quantize.get_next_autoregressive_input(si, len(self.raw_scale_schedule), beam_accu_BChw, h_BChw)

                    beam_last_stage = beam_last_stage.view(B, vae.Cvae, -1).transpose(1, 2)
                    beam_last_stage = self.word_embed(self.norm0_ve(beam_last_stage))
                    beam_last_stage = beam_last_stage.repeat(bs//B, 1, 1)
                    
                    
                    beam_logits_BlV = self.get_scale_logits(si=nex_is, last_stage=beam_last_stage, cond_BD_or_gss=cond_BD_or_gss, \
                            ca_kv = ca_kv, cond_BD=cond_BD, scale_schedule=scale_schedule, \
                            B=B, need_to_pad=need_to_pad, attn_fn=attn_fn, cache_now=False)

                    # 不同batch的结果应该不一样
                    beam_probs = F.softmax(beam_logits_BlV, dim=-1)
                    if score_compare == "max":
                        beam_value_max_Bl, beam_idx_max_Bl = beam_probs.max(dim=-1)
                        beam_score = beam_value_max_Bl.sum(dim=-1)
                    elif score_compare == "max_2max":
                        top2_values, top2_indices = beam_probs.topk(2, dim=-1)
                        prob_max_2max = 2 * top2_values[...,0] - top2_values[...,1]
                        beam_score = prob_max_2max.sum(dim=-1)

                    for _b in range(B):
                        if beam_find_best_score_Bl[_b] < (beam_score[_b] + beam_choose_p[_b]):
                            beam_find_best_idx_Bl[_b] = beam_idx_Bl[_b]
                            beam_find_best_score_Bl[_b] = (beam_score[_b] + beam_choose_p[_b])

                
                idx_Bl = beam_find_best_idx_Bl


            h_BChw = vae.quantize.embedding(idx_Bl).float()   # BlC
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][1], scale_schedule[si][2])
            
            ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
            idx_Bl_list.append(idx_Bl)
            
            
            accu_BChw, last_stage = vae.quantize.get_next_autoregressive_input(si, len(self.raw_scale_schedule), accu_BChw, h_BChw)
            
            if si != num_stages_minus_1:
                last_stage = last_stage.view(B, vae.Cvae, -1).transpose(1, 2)
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)
                
                
                
        for b in self.unregistered_blocks: 
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)


        if not ret_img:
            return ret, idx_Bl_list, []

        img = vae.fhat_to_img(accu_BChw)
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
        return ret, idx_Bl_list, img
    
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
        self, label_B_or_BLT: Tuple[torch.FloatTensor, torch.IntTensor, int], 
        x_BLC_wo_prefix: torch.Tensor,
        scale_schedule:List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        # if cfg_infer:
        #     return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]
        
        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            if(len(kv_compact.shape) == 3): # B L C -> B*L C
                kv_compact = kv_compact.reshape(-1,kv_compact.shape[-1])
            # drop cond
            total = 0
            if not cfg_infer:
                for le in lens:
                    if random.random() < self.cond_drop_rate:
                        kv_compact[total:total+le] = self.cfg_uncond[:le]
                    total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.low_norm(kv_compact).contiguous()
            sos = cond_BD = self.low_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.low_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
        
            # with open('log.txt', 'a') as f:
            #     f.write(f'sos:{sos.unsqueeze(1).expand(B, 1, -1)}\n')
            #     f.write(f'sos:{self.pos_start.expand(B, 1, -1)}\n')        
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)
            
            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
                 

                
            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None
        # [2. block loop]
        SelfAttnBlock.forward, CrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    # x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    # x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    # x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    # x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size    
        
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
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
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
        for m in self.modules():
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
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

if __name__ == "__main__":
    
    
    vqvae = VQVAE().cuda()
    low_channel = 32
    var = SRVAR(vqvae,
                low_channel=low_channel,
                low_len=512,
                block_chunks = 4,
                # pad_to_multiplier=128,
                # use_flex_attn=True,
                pn="1M",
                rope2d_normalized_by_hw = 2,
                checkpointing = "full-block",
                
                
                ).cuda()
    # var = SRVAR(vqvae,low_channel=2048,low_len=512,block_chunks = 4).cuda()
    
    
    tini: float = 0.02     
    diva: int = 1                       # rescale_attn_fc_weights
    hd0: float = 0.02  
    aln: float = 1e-3                   # multiplier of ada_lin.w's initialization
    alng: float = -1 
    var.init_weights(other_std=tini)
    var.special_init(aln_init=aln, aln_gamma_init=alng, scale_head=hd0, scale_proj=diva)

    
    import os
    import torch.distributed as tdist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    tdist.init_process_group("nccl", rank=0, world_size=1)

    inp_B3HW = torch.randn(3, 3, 256, 256).cuda()
    
    gt_idx_Bl= vqvae.img_to_idxBl(inp_B3HW)
    gt_BL = torch.cat(gt_idx_Bl, dim=1)
    x_BLCv_wo_first_l = vqvae.quantize.idxBl_to_var_input(gt_idx_Bl)
    print(x_BLCv_wo_first_l.shape)
    
    lens = torch.tensor([3, 5, 2],dtype=torch.int32).cuda()  # 每个句子的 token 长度
    max_seqlen_k = lens.max().cuda()  # 5
    cu_seqlens_k = torch.cumsum(torch.cat([torch.tensor([0],dtype=torch.int32).cuda(), lens]), dim=0).cuda()
    
    cu_seqlens_k = cu_seqlens_k.to(dtype = torch.int32)

    label_B_or_BLT = (torch.randn(sum(lens), low_channel).cuda(), lens, cu_seqlens_k, max_seqlen_k)
    
    B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
    T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
    V = vqvae.vocab_size
    device = inp_B3HW.device

    h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
    h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template]["1M"]['scales']
    scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
    # raw_scale_schedule = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    # scale_schedule = [(1,h,h) for h in raw_scale_schedule]
    
    # logits = var(label_B_or_BLT, x_BLCv_wo_first_l, scale_schedule)
    # print(logits.shape)
    # logits.mean().backward()
    _,_,image_list = var.autoregressive_infer_cfg(
        B = 3,
        vae=vqvae,
        label_B_or_BLT=label_B_or_BLT, 
        scale_schedule=scale_schedule,
        cfg_list=[1.0]*len(scale_schedule),tau_list=[1.0]*len(scale_schedule),
        ret_img = True,
        top_k=1,top_p=1.0,
        inference_mode=True
    )
    from PIL import Image
    Image.fromarray(image_list[0].detach().cpu().numpy()).save("test.png")
    
