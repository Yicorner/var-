# %%
import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
import torch.distributed as tdist
import torch
from PIL import Image
from torch.utils.data import DataLoader
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import numpy as np
import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
import math
from models import SRVAR, VQVAE, build_vae_srvar

args: arg_util.Args = arg_util.Args()
# args.data = "./data/df2k_ost/GT_resized"
# args.data = "./data/DIV2K_train_HR"
# args.data = "./data/train"
# args.data = "./data/brats_256_t1_new/train"
args.batch_size = 1
args.data_path = "../VAR/data/brats_256_t1_pair/"
args.fp16=1
args.alng = 1e-3
args.wpe = 0.1
args.pn = "1M"
args.rope2d_normalized_by_hw = 2
args.rope2d_each_sa_layer = 1
args.checkpointing = "full-block"
args.vae_ckpt = "ckpt-150.pth"
args.Ct5 = 32
args.tlen = 1024
args.top_k = 900
args.top_p = 0.95
args.device = "cuda"

# %%


V=args.vocab_size
Cvae=args.Ct5
ch=160
share_quant_resi=4
patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)   # 10 steps by default

# print(torch.load(ckpt_path, map_location='cpu')['trainer'].keys())
vae = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(args.device)
    
srvar_kw = dict(
    low_channel=args.Ct5, low_len=args.tlen,
    norm_eps=args.norm_eps, rms_norm=args.rms,
    shared_aln=args.saln, head_aln=args.haln,
    cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
    cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau, cos_attn=args.cos, swiglu=args.swi,
    raw_scale_schedule=patch_nums,
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
if args.dp >= 0: srvar_kw['drop_path_rate'] = args.dp
if args.hd > 0: srvar_kw['num_heads'] = args.hd

# print(f'[create srvar] constructor kw={srvar_kw}\n')

ckpt_path_srvar = f"local_output/ar-ckpt-last.pth"
vae.load_state_dict(torch.load(ckpt_path_srvar, map_location='cpu')['trainer']['vae_local'])

srvar_kw['vae_local'] = vae


srvar: SRVAR = SRVAR(**srvar_kw)
srvar = srvar.to(args.device)
srvar.load_state_dict(torch.load(ckpt_path_srvar, map_location='cpu')['trainer']['srvar_wo_ddp'])

# %%
dataset_train, dataset_val = build_dataset(
    args.data_path, final_reso=256, hflip=args.hflip, mid_reso=1.25,
)
types = str((type(dataset_train).__name__, type(dataset_val).__name__))

ld_val = DataLoader(
    dataset_val, num_workers=args.workers, batch_size=args.batch_size,shuffle=False,
)
del dataset_val

ld_train = DataLoader(
    dataset=dataset_train, num_workers=args.workers,batch_size=args.batch_size,shuffle=False,
)
srvar = srvar.eval()
vae = vae.eval()
# %%
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12115'
#     tdist.init_process_group("nccl", rank=rank, world_size=world_size)
# setup(0,1)

# %%
tot = 0

temp = 0
for inp_B3HW_low, inp_B3HW_super in ld_train:
    
    inp_B3HW_low = inp_B3HW_low.to(dist.get_device(), non_blocking=True)
    inp_B3HW_super = inp_B3HW_super.to(dist.get_device(), non_blocking=True)
    
    B, V = inp_B3HW_low.shape[0], vae.vocab_size
    gt_idx_Bl_low = vae.img_to_idxBl(inp_B3HW_low)
    gt_BL_low = torch.cat(gt_idx_Bl_low, dim=1)
    x_BLCv_wo_first_l_low = vae.quantize.embedding(gt_BL_low)  #这里应该是gt的idx组成的embedding
    
    
    gt_idx_Bl_super= vae.img_to_idxBl(inp_B3HW_super)
    gt_BL_super = torch.cat(gt_idx_Bl_super, dim=1)
    x_BLCv_wo_first_l_super= vae.quantize.idxBl_to_var_input(gt_idx_Bl_super)
    
    # [3,679,32]
    lowLen, lowC = x_BLCv_wo_first_l_low.shape[1], x_BLCv_wo_first_l_low.shape[2]
    
    lens = torch.tensor([lowLen] * B,dtype=torch.int32).to(device=x_BLCv_wo_first_l_super.device)  # 每个句子的 token 长度
    
    x_BLCv_wo_first_l_low = x_BLCv_wo_first_l_low.reshape( -1, lowC)
    
    max_seqlen_k = lens.max().to(device=x_BLCv_wo_first_l_super.device)  # 5
    cu_seqlens_k = torch.cumsum(torch.cat([torch.tensor([0],dtype=torch.int32).to(device=x_BLCv_wo_first_l_super.device), lens]), dim=0).to(device=x_BLCv_wo_first_l_super.device).to(dtype = torch.int32)
    label_B_or_BLT = (x_BLCv_wo_first_l_low, lens, cu_seqlens_k, max_seqlen_k)
    
    
    h_div_w = inp_B3HW_low.shape[-2] / inp_B3HW_low.shape[-1]
    T = 1 if inp_B3HW_low.dim() == 4 else inp_B3HW_low.shape[2]
    h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template]["1M"]['scales']
    scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
    
    
    idx_predict = []
    for si,pn in enumerate(patch_nums):
        idx_predict.append(torch.zeros(B,pn*pn,dtype=torch.int64).to(device=x_BLCv_wo_first_l_super.device))
    print(gt_idx_Bl_super[1].dtype)
    
    Cul_L = 0
    for si, pn in enumerate(patch_nums):
        num_pn = pn*pn
        temp_BLC = vae.quantize.idxBl_to_var_input(idx_predict)
        logits_BLV = srvar(label_B_or_BLT, temp_BLC ,scale_schedule)
        idx_train = logits_BLV.data.argmax(dim=-1)
        
        idx_predict[si] = idx_train[:,Cul_L:Cul_L+num_pn].reshape(B,num_pn)
        Cul_L = Cul_L + num_pn
        print(f"si:{si} done!")
    
    idx_Bl_list = idx_predict
    idx_predict = torch.cat(idx_predict,dim=1)

    logits_BLV = srvar(label_B_or_BLT, x_BLCv_wo_first_l_super,scale_schedule)
    idx_train = logits_BLV.data.argmax(dim=-1)

    show = 14+16
    print("test :",idx_predict[0][:show])
    #print(idx_Bl_list[0][0],idx_Bl_list[1][0],idx_Bl_list[2][0])
    print("train:",idx_train[0][:show])
    print("gt   :",gt_BL_super[0][:show])
    
    print("acc_train:",(idx_train == gt_BL_super).sum().item() / (B*gt_BL_super.shape[1]))
    print("acc_list:",(idx_predict == gt_BL_super).sum().item() / (B*gt_BL_super.shape[1]))
    print("train_list:",(idx_predict == idx_train).sum().item() / (B*gt_BL_super.shape[1]))


    nup_test = vae.idxBl_to_img(idx_Bl_list, same_shape=True, last_one=False)
    nup_gt = vae.idxBl_to_img(gt_idx_Bl_super, same_shape=True, last_one=False)
    start_idx = 0
    result_train = []
    for path_num in patch_nums:
        end_idx = start_idx + path_num ** 2
        result_train.append(idx_train[:, start_idx:end_idx])  
        start_idx = end_idx  
    nup_train = vae.idxBl_to_img(result_train, same_shape=True, last_one=False)
    
    def process_image(x):
        """处理单张图片"""    
        x = x[0].detach().cpu().permute(1, 2, 0).numpy()  # 转换为 HWC 格式的 numpy 数组
        x = (x * 0.5 + 0.5) * 255  # 反归一化并缩放到 [0, 255]
        x = x.astype(np.uint8)  # 转换为 uint8
        return x

    def concatenate_images(images, axis):
        """沿指定轴拼接图片"""
        return np.concatenate(images, axis=axis)
    
    train_combined = concatenate_images([process_image(x) for x in nup_train], axis=1)  # 左右连接
    test_combined = concatenate_images([process_image(x) for x in nup_test], axis=1)   # 左右连接
    gt_combined = concatenate_images([process_image(x) for x in nup_gt], axis=1)      # 左右连接

    # 将三个结果上下连接
    final_image = concatenate_images([train_combined, test_combined, gt_combined], axis=0)  # 上下连接
    Image.fromarray(final_image).save("combined_image.png")





