import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist


class Args(Tap):
    use_ref: bool = False
    use_diff: bool = True            # always True now (continuous DiffLoss head); kept for ckpt compat
    zero: int = 0                       # ds zero
    data_path: str = '/path/to/imagenet'
    enable_checkpointing: str = None    # checkpointing strategy: full-block, self-attn
    pad_to_multiplier: int = 1          # >1 for padding the seq len to a multiplier of this
    train_h_div_w_list: list = [1.0]
    val_and_saving_per_ep: int = 5
    use_are_loss_weight: bool = False
    # VAE
    vocab_size: int = 0                 # unused for continuous VAE; kept for ckpt compat
    vae_ckpt: str = None
    vae_ch: int = 128                   # HR VAE base channel; must match stage2 ckpt
    quant_resi: float = 0.5             # quant_resi ratio; must match stage2 ckpt
    share_quant_resi: int = 4           # quant_resi share mode; must match stage2 ckpt
    vfast: int = 0
    # LR conditioning
    lr_folder: str = 'LR_64x64'         # subdir under DATA_PATH/{train,val} for LR images
    hr_folder: str = 'HR'               # subdir under DATA_PATH/{train,val} for HR images
    lr_cond_source: str = 'srvar_encoder'  # 'srvar_encoder' or 'lr_vae'
    stage1_ckpt: str = ''               # non-empty -> build & load LR_VAE
    skip_scale0_loss: bool = False      # plan-A only: drop scale[0] from DiffLoss target/z
    same_shape: bool = False            # if True, LR is bicubic-resized to HR size (legacy)
    # DiffLoss head
    diffloss_w: int = 1024
    diffloss_d: int = 3
    diff_steps: str = '100'             # inference sampling steps (spaced IDDPM)
    diffloss_batch_mul: int = 4         # per-token MAR-style batch multiplier
    cfg_infer: float = 1.0              # inference CFG scale (>1 sharpens condition)
    # Reconstruction visualization / metadata
    save_reconstruction_images: bool = True
    reconstruction_save_interval: int = 0   # 0: follow train log iters; >0: every N iters
    reconstruction_max_samples: int = 4
    reconstruction_dir_name: str = 'reconstruction_samples'
    record_reconstruction_metadata: bool = True
    eval_ar_max_batches: int = 4
    train_log_points_per_epoch: int = 8
    log_train_psnr: bool = False
    # VAR
    # depth: int = 16     # VAR depth
    # ini: float = -1     # -1: automated model parameter initialization
    
    sche: str = 'lin0'      # lr schedule
    anorm: bool = True      # whether to use L2 normalized attention
    fuse: bool = True       # whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc.
    
    # data
    patch_size: int = 16
    # patch_nums: tuple = (1,3,5,8,12,16)    # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) 
    
    hflip: bool = False         # augmentation: horizontal flip
    
    # progressive training
    pg: float = 0.0         # >0 for use progressive training during [0%, this] of training
    pg0: int = 4            # progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc
    pgwp: float = 0         # num of warmup epochs at each progressive stage
    
    # GPT
    block_chunks: int = 4

    tfast: int = 0                      # compile GPT
    rms: bool = False
    aln: float = 1                   # multiplier of ada_lin.w's initialization
    alng: float = 5e-6                  # the multiplier of ada_lin.w[gamma channels]'s initialization
    saln: bool = False                  # whether to use a shared adaln layer
    haln: bool = True                   # whether to use a specific adaln layer in head layer
    nm0: bool = False                   # norm before word proj linear
    tau: float = 1                      # tau of self attention in GPT
    cos: bool = True                    # cosine attn as in swin v2
    swi: bool = False                   # whether to use FFNSwiGLU, instead of vanilla FFN
    dp: float = -1
    drop: float = 0.0                   # GPT's dropout (VAE's is --vd)
    hd: int = 0
    ca_gamma: float = -1                # >=0 for using layer-scale for cross attention
    diva: int = 1                       # rescale_attn_fc_weights
    hd0: float = 0.02                   # head.w *= hd0
    dec: int = 1                        # dec depth
    cum: int = 3                        # cumulating fea map as GPT TF input, 0: not cum; 1: cum @ next hw, 2: cum @ final hw

    tp: float = 0.0                     # top-p
    tk: float = 0.0                     # top-k
    tini: float = 0.02                  # init parameters
    cfg: float = 0.1                    # >0: classifier-free guidance, drop cond with prob cfg
    rand_uncond = False                 # whether to use random, unlearnable uncond embeding
    fp16: int = 0                       # 1: fp16, 2: bf16, >2: fp16's max scaling multiplier todo: 记得让quantize相关的feature都强制fp32！另外residueal最好也是fp32（根据flash-attention）nn.Conv2d有一个参数是use_float16？
    fuse: bool = False                  # whether to use fused mlp
    fused_norm: bool = False            # whether to use fused norm
    flash: bool = False                 # whether to use customized flash-attn kernel
    use_flex_attn: bool = False         # whether to use flex_attn to speedup training
    stable: bool = False
    tblr: float = 6e-4
    tlr: float = None                   # vqgan: 4e-5
    twd: float = 0.005                  # vqgan: 0.01
    twde: float = 0
    ls: float = 0.0                     # label smooth
    ep: int = 100
    wp: float = 0
    wp0: float = 0.005
    wpe: float = 0.3                    # 0.001, final cosine lr = wpe * peak lr
    tclip: float = 2.                   # <=0 for not grad clip GPT; >100 for per-param clip (%= 100 automatically)
    cdec: bool = False                  # decay the grad clip thresholds of GPT and GPT's word embed
    opt: str = 'adamw'                  # lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5（比Adam学习率低四倍）和wd=0.8（比Adam高八倍）；比如在小的 batch_size 时，Lion 的表现不如 AdamW
    ada: str = ''                       # adam's beta0 and beta1 for VAE or GPT, '0_0.99' from style-swin and magvit, '0.5_0.9' from VQGAN
    oeps: float = 0                     # adam's eps, pixart uses 1e-10
    afuse: bool = True                  # fused adam

    # data
    pn: str = ''                        # pixel nums, choose from 0.06M, 0.25M, 1M
    scale_schedule: tuple = None        # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    # patch_size: int = None              # [automatically set; don't specify this] = 2 ** (len(args.scale_schedule) - 1)
    resos: tuple = None                 # [automatically set; don't specify this]
    workers: int = 0                    # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    lbs: int = 0                        # local batch size; if lbs != 0, bs will be ignored, and will be reset as round(args.lbs / args.ac) * dist.get_world_size()
    bs: int = 0                         # global batch size; if lbs != 0, bs will be ignored
    batch_size: int = 0                 # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size())
    glb_batch_size: int = 0             # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    ac: int = 1                         # gradient accumulation
    r_accu: float = 1.0                 # [automatically set; don't specify this] = 1 / args.ac
    norm_eps: float = 1e-6              # norm eps for infinity
    tlen: int = 512                     # truncate text embedding to this length
    Ct5: int = 2048                     # feature dimension of text encoder

    rope2d_each_sa_layer: int = 0       # apply rope2d to each self-attention layer
    rope2d_normalized_by_hw: int = 1    # apply normalized rope2d
    add_lvl_embeding_only_first_block: int = 1 # apply lvl pe embedding only first block or each block
    
    #TODO Neesky:不知道干什么，目前看来和flex_attn有关
    always_training_scales: int = 100   # trunc training scales
    apply_spatial_patchify: int = 0     # apply apply_spatial_patchify or not
    debug_bsc: int = 0                  # save figs and set breakpoint for debug bsc and check input
    
    # would be automatically set in runtime
    cmd: str = ' '.join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]' # [automatically set; don't specify this]
    commit_id: str = subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()    # [automatically set; don't specify this]
    acc_mean: float = None      # [automatically set; don't specify this]
    acc_tail: float = None      # [automatically set; don't specify this]
    L_mean: float = None        # [automatically set; don't specify this]
    L_tail: float = None        # [automatically set; don't specify this]
    vacc_mean: float = None     # [automatically set; don't specify this]
    vacc_tail: float = None     # [automatically set; don't specify this]
    vL_mean: float = None       # [automatically set; don't specify this]
    vL_tail: float = None       # [automatically set; don't specify this]
    grad_norm: float = None     # [automatically set; don't specify this]
    cur_lr: float = None        # [automatically set; don't specify this]
    cur_wd: float = None        # [automatically set; don't specify this]
    cur_it: str = ''            # [automatically set; don't specify this]
    cur_ep: str = ''            # [automatically set; don't specify this]
    remain_time: str = ''       # [automatically set; don't specify this]
    finish_time: str = ''       # [automatically set; don't specify this]
    
    # environment
    local_out_dir_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # [automatically set; don't specify this]
    tb_log_dir_path: str = '...tb-...'  # [automatically set; don't specify this]
    log_txt_path: str = '...'           # [automatically set; don't specify this]
    last_ckpt_path: str = '...'         # [automatically set; don't specify this]
    
    tf32: bool = True       # whether to use TensorFloat32
    device: str = 'cpu'     # [automatically set; don't specify this]
    seed: int = None        # seed
    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    same_seed_for_all_ranks: int = 0     # this is only for distributed sampler
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg: bool = 'KEVIN_LOCAL' in os.environ       # only used when debug about unused param in DDP
    dbg_nan: bool = False   # 'KEVIN_LOCAL' in os.environ
    
    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep: # first time to dump log
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(),  'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch, 'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')
        
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'
    
    @property
    def gpt_training(self):
        return len(self.model) > 0

def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1
    else:
        if args.data_path == '/path/to/imagenet':
            raise ValueError(f'{"*"*40}  please specify --data_path=/path/to/imagenet  {"*"*40}')
    
    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    # init torch distributed
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)
    
    # set env
    args.set_tf32(args.tf32)
    if args.dbg:
        torch.autograd.set_detect_anomaly(True)
    args.seed_everything(benchmark=args.pg == 0)
    
    # update args: data loading
    args.device = dist.get_device()
    # CLI `--patch_nums 1 2 3 ...` is parsed as strings by Tap; coerce once here.
    args.patch_nums = tuple(int(p) for p in args.patch_nums)
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    
    # update args: bs and lr
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size() # bs为一轮的bs，原先的是附加上ac的bs
    args.workers = min(max(0, args.workers), args.batch_size)
    
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.twde = args.twde or args.twd 
    
    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}."
    
    if args.wp == 0:
        args.wp = args.ep * 1/50
    
    # update args: progressive training
    if args.pgwp == 0:
        args.pgwp = args.ep * 1/300
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'
    
    # update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args
