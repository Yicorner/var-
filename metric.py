# %%
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
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
from torchvision.transforms import transforms
import pyiqa
from skimage import io
import glob

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr_metric = pyiqa.create_metric('psnr', device=device)
ssim_metric = pyiqa.create_metric('ssim', device=device)
fid_metric = pyiqa.create_metric('fid', device=device)
maniqa_metric = pyiqa.create_metric('maniqa', device=device)
lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
dists_iqa_metric = pyiqa.create_metric('dists', device=device)
niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

            
def process_image(x):
    """处理单张图片"""    
    x = x.detach().cpu().permute(0, 2, 3, 1).numpy()  # 转换为 HWC 格式的 numpy 数组
    x = (x * 0.5 + 0.5) * 255  # 反归一化并缩放到 [0, 255]
    x = x.astype(np.uint8)  # 转换为 uint8
    return x
def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    tdist.init_process_group("nccl", rank=rank, world_size=world_size)
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def get_value(x):
    return x.item() if isinstance(x, torch.Tensor) else x
def write_metrics_to_file(filename, metric_name, values, outCMD =False):
    mean_val = sum(values) / len(values)
    max_val = max(values)
    min_val = min(values)
    if( outCMD):
        print(f"{metric_name}: Mean = {get_value(mean_val)}, Max = {get_value(max_val)}, Min = {get_value(min_val)}")
    with open(filename, "a") as f:
        f.write(f"{metric_name}: Mean = {get_value(mean_val)}, Max = {get_value(max_val)}, Min = {get_value(min_val)}\n")

def get_img(args, ld_val, maxtot, ckpt_paths, beam_search_nums, choose_min,score_compare):
    
    V=args.vocab_size
    Cvae=args.Ct5
    ch=160
    share_quant_resi=4
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16) 

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
    for ckpt_path_srvar in ckpt_paths:
        vae.load_state_dict(torch.load(ckpt_path_srvar, map_location='cpu')['trainer']['vae_local'])
        srvar_kw['vae_local'] = vae
        srvar: SRVAR = SRVAR(**srvar_kw)
        srvar = srvar.to(args.device)
        srvar.load_state_dict(torch.load(ckpt_path_srvar, map_location='cpu')['trainer']['srvar_wo_ddp'])

        srvar.eval()
        vae.eval()
        
        out_dir = os.path.join("metric_results",os.path.basename(ckpt_path_srvar))
        predict_dir = os.path.join(out_dir,"predict")
        gt_dir = os.path.join(out_dir,"gt")
        os.makedirs(predict_dir,exist_ok=True)
        os.makedirs(gt_dir,exist_ok=True)
        
        for idx, (inp_B3HW_low, inp_B3HW_super) in tqdm(enumerate(ld_val), total=len(ld_val)):
            if(idx>=maxtot and maxtot>0):
                break
            
            inp_B3HW_low = inp_B3HW_low.to(dist.get_device(), non_blocking=True)
            inp_B3HW_super = inp_B3HW_super.to(dist.get_device(), non_blocking=True)
            
            B, V = inp_B3HW_low.shape[0], vae.vocab_size
            
            low_f = vae.img_to_f(inp_B3HW_low)
            low_f = low_f.permute(0, 2, 3, 1)
            low_f = low_f.reshape(B, low_f.shape[1] * low_f.shape[2], low_f.shape[3])# B=36
            
            gt_idx_Bl_super= vae.img_to_idxBl(inp_B3HW_super)
            gt_BL_super = torch.cat(gt_idx_Bl_super, dim=1)
            x_BLCv_wo_first_l_super= vae.quantize.idxBl_to_var_input(gt_idx_Bl_super)

            lowLen, lowC = low_f.shape[1], low_f.shape[2]
            lens = torch.tensor([lowLen] * B,dtype=torch.int32).to(device=x_BLCv_wo_first_l_super.device)  # 每个句子的 token 长度
            low_f = low_f.reshape( -1, lowC)
            max_seqlen_k = lens.max().to(device=x_BLCv_wo_first_l_super.device)  # 5
            cu_seqlens_k = torch.cumsum(torch.cat([torch.tensor([0],dtype=torch.int32).to(device=x_BLCv_wo_first_l_super.device), lens]), dim=0).to(device=x_BLCv_wo_first_l_super.device).to(dtype = torch.int32)
            label_B_or_BLT = (low_f, lens, cu_seqlens_k, max_seqlen_k)
            
            h_div_w = inp_B3HW_low.shape[-2] / inp_B3HW_low.shape[-1]
            T = 1 if inp_B3HW_low.dim() == 4 else inp_B3HW_low.shape[2]
            h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template]["1M"]['scales']
            scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
            
            # idx_predict = []
            # for si,pn in enumerate(patch_nums):
            #     idx_predict.append(torch.zeros(B,pn*pn,dtype=torch.int64).to(device=inp_B3HW_low.device))
            
            # Cul_L = 0
            # for si, pn in enumerate(patch_nums):
            #     num_pn = pn*pn
            #     temp_BLC = vae.quantize.idxBl_to_var_input(idx_predict)
            #     logits_BLV = srvar(label_B_or_BLT, temp_BLC ,scale_schedule,cfg_infer=True)
            #     idx_temp = logits_BLV.data.argmax(dim=-1)
                
            #     idx_predict[si] = idx_temp[:,Cul_L:Cul_L+num_pn].reshape(B,num_pn)
            #     Cul_L = Cul_L + num_pn
                
            # idx_Bl_list = idx_predict
            # idx_predict = torch.cat(idx_predict,dim=1)
            
            # nup_test = vae.idxBl_to_img(idx_Bl_list, same_shape=True, last_one=True)
            # nup_test = process_image(nup_test)

            ret, idx_Bl_list, img = srvar.autoregressive_infer_cfg(vae=vae, label_B_or_BLT=label_B_or_BLT, 
                                scale_schedule=scale_schedule,
                                ret_img=True,
                                B=B,
                                choose_min=choose_min,
                                score_compare=score_compare,
                                beam_search_nums = beam_search_nums
                                )

            nup_test = img.detach().cpu().numpy()


            nup_gt = process_image(inp_B3HW_super)

            for i in range(B):
                _data = nup_gt[i]
                _rec_B3HW = nup_test[i]
                Image.fromarray(_rec_B3HW).save(os.path.join(predict_dir,f"{idx*B+i}.png"))
                Image.fromarray(_data).save(os.path.join(gt_dir,f"{idx*B+i}.png"))

    

    
def metric(metric_path,ckpt_paths,beam_search_nums,choose_min,score_compare):
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    for fold in ckpt_paths:
        fold = os.path.basename(fold)
        print(f"now {fold}")
        out_dir = os.path.join("metric_results",fold)
        predict_dir = os.path.join(out_dir,"predict")
        gt_dir = os.path.join(out_dir,"gt")
        
        gt_img_paths = []

        psnr_folder = []
        ssim_folder = []
        lpips_score = []
        dists_score = []
        niqe_score = []
        lpips_iqa = []
        musiq_iqa = []
        maniqa_iqa = []
        clip_iqa = []
        gt_img_paths.extend(sorted(glob.glob(f'{gt_dir}/*.png'))[:])
        
        
        for gt_img_path in tqdm(gt_img_paths):
            GT_image = img_preproc(Image.open(gt_img_path).convert('RGB'))
            prediction_img_path = gt_img_path.replace("/gt/", "/predict/")
            VARPrediction_img = img_preproc(Image.open(prediction_img_path).convert('RGB'))
        
            img1 = rgb2ycbcr_pt(img2tensor(io.imread(gt_img_path)),  y_only=True).to(torch.float64)
            img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)),  y_only=True).to(torch.float64)
            img1 = torch.squeeze(img1)
            img2 = torch.squeeze(img2)
            
            ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_img_path))
            clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            maniqa_iqa.append(maniqa_metric(prediction_img_path))
            dists_score.append(dists_iqa_metric(prediction_img_path, gt_img_path))
            niqe_score.append(niqe_iqa_metric(prediction_img_path))

        with open(metric_path, "a") as f:
            f.write(f"fold = {fold}\n")
        write_metrics_to_file(metric_path, "PSNR", psnr_folder)
        write_metrics_to_file(metric_path, "SSIM", ssim_folder)
        write_metrics_to_file(metric_path, "LPIPS", lpips_iqa)
        write_metrics_to_file(metric_path, "DISTS", dists_score)
        write_metrics_to_file(metric_path, "NIQE", niqe_score)
        write_metrics_to_file(metric_path, "CLIP-IQA", clip_iqa)
        write_metrics_to_file(metric_path, "MUSIQ", musiq_iqa)
        write_metrics_to_file(metric_path, "MANIQA", maniqa_iqa)
        print(f"now fid")
        fid_value = fid_metric(gt_dir, predict_dir)
        with open(metric_path, "a") as f:
            f.write(f"FID = {get_value(fid_value)}\n")
            
            
if __name__ == "__main__":
    args: arg_util.Args = arg_util.Args()

    args.batch_size = 24
    args.fp16=1
    args.alng = 1e-3
    args.wpe = 0.1
    args.pn = "1M"
    args.rope2d_normalized_by_hw = 2
    args.rope2d_each_sa_layer = 1
    args.enable_checkpointing = "full-block"
    args.tlen = 1024
    args.device = "cuda"

    args.Ct5 = 32
    args.vocab_size = 4096
    args.data_path = "./data/brats_256_t1_2021_pair_4x/"
    maxtot = -1
    out_path = "./metric.txt"


    beam_search_nums = 0
    choose_min = "max"
    score_compare = "max"

    args.seed = 666
    args.seed_everything(False)


    dataset_train, dataset_val = build_dataset(
        args.data_path,augment=False
    )
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))

    ld_val = DataLoader(
        dataset_val, num_workers=args.workers, batch_size=args.batch_size,shuffle=True,
    )


    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers,batch_size=args.batch_size,shuffle=False,
    )
    del dataset_val,dataset_train
    
    # ckpt_paths = [f"local_output/ckpt-{i}.pth" for i in range(252,281,3)]
    ckpt_paths = [f"local_output/ckpt-{321}.pth"]
    print(ckpt_paths)

    get_img(args = args,ld_val = ld_val ,maxtot = maxtot,ckpt_paths = ckpt_paths,
                        beam_search_nums = beam_search_nums,
                        choose_min = choose_min,
                        score_compare = score_compare)
    metric(out_path,ckpt_paths ,beam_search_nums = beam_search_nums,choose_min = choose_min,score_compare = score_compare)

