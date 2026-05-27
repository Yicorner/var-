import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader
# TODO:Neesky 如果使用flex_attention
# torch._dynamo.config.optimize_ddp=False

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
import math

from torch.nn.parallel import DistributedDataParallel as DDP
from models import SRVAR, Stage3Scale0Encoder, VQVAE, build_vae_srvar, build_lr_vae, LR_VAE
from SRtrainer import SRVARTrainer
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params

warnings.filterwarnings("ignore", category=FutureWarning)


def _read_ckpt_arg(ckpt: dict, key: str, default):
    args_state = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    if isinstance(args_state, dict):
        return args_state.get(key, default)
    return default


def _read_stage3_ckpt_config(ckpt: dict, key: str, default):
    trainer = ckpt.get('trainer', {}) if isinstance(ckpt, dict) else {}
    config = trainer.get('config', {}) if isinstance(trainer, dict) else {}
    if isinstance(config, dict) and key in config:
        return config.get(key, default)
    return _read_ckpt_arg(ckpt, key, default)


def _extract_stage3_state(ckpt: dict) -> dict:
    trainer = ckpt.get('trainer', {}) if isinstance(ckpt, dict) else {}
    if isinstance(trainer, dict) and isinstance(trainer.get('stage3_wo_ddp'), dict):
        return trainer['stage3_wo_ddp']
    for key in ('stage3_wo_ddp', 'state_dict'):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            return ckpt[key]
    raise KeyError('Could not locate stage3 weights in checkpoint.')


def _build_stage3_encoder(args: arg_util.Args, vae_local: VQVAE) -> Optional[Stage3Scale0Encoder]:
    if args.scale0_start_source != 'stage3':
        return None
    assert args.stage3_ckpt, '--stage3_ckpt is required when --scale0_start_source=stage3.'
    ckpt = torch.load(args.stage3_ckpt, map_location='cpu')
    latent_size = int(_read_stage3_ckpt_config(ckpt, 'stage3_latent_size', args.stage3_latent_size))
    assert latent_size == int(args.stage3_latent_size), (
        f'stage3 ckpt latent_size={latent_size} != --stage3_latent_size={args.stage3_latent_size}'
    )
    assert int(vae_local.quantize.v_patch_nums[0]) == latent_size, (
        f'vae scale[0]={vae_local.quantize.v_patch_nums[0]} != stage3 latent_size={latent_size}; '
        f'check --patch_nums and --stage3_latent_size.'
    )

    cvae = int(_read_stage3_ckpt_config(ckpt, 'Cvae', _read_stage3_ckpt_config(ckpt, 'vocab_width', args.Ct5)))
    ch = int(_read_stage3_ckpt_config(ckpt, 'ch', args.vae_ch))
    img_channels = int(_read_stage3_ckpt_config(ckpt, 'img_channels', args.img_channels))
    quant_conv_ks = int(_read_stage3_ckpt_config(ckpt, 'quant_conv_ks', getattr(vae_local, 'quant_conv_ks', 3)))
    dropout = float(_read_stage3_ckpt_config(ckpt, 'dropout', getattr(vae_local, 'dropout', 0.0)))

    assert cvae == int(vae_local.Cvae), f'stage3 Cvae={cvae} != VAE Cvae={vae_local.Cvae}'
    assert img_channels == int(args.img_channels), (
        f'stage3 img_channels={img_channels} != args.img_channels={args.img_channels}'
    )

    stage3 = Stage3Scale0Encoder(
        z_channels=cvae,
        ch=ch,
        dropout=dropout,
        quant_conv_ks=quant_conv_ks,
        latent_size=latent_size,
        img_channels=img_channels,
    ).to(dist.get_device())
    stage3.load_state_dict(_extract_stage3_state(ckpt), strict=True)
    stage3.eval()
    for p in stage3.parameters():
        p.requires_grad_(False)
    print(f'[stage3_ckpt] loaded frozen Stage3Scale0Encoder from {args.stage3_ckpt}')
    return stage3

def build_everything(args: arg_util.Args):
    # resume: --resume takes precedence; else fall back to latest ckpt under BED when auto_resume=True
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # =============== build logger ===============
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    assert args.vae_ckpt, '--vae_ckpt must be provided (myvaex stage2 ckpt path).'
    assert args.scale0_start_source in ('transformer', 'stage3'), \
        f"--scale0_start_source must be 'transformer' or 'stage3', got {args.scale0_start_source!r}"
    assert args.stage3_context_mode in ('both', 'prefix_only'), \
        f"--stage3_context_mode must be 'both' or 'prefix_only', got {args.stage3_context_mode!r}"
    args.stage3_latent_size = int(args.stage3_latent_size)
    if args.scale0_start_source == 'stage3':
        assert args.stage3_ckpt, '--stage3_ckpt is required when --scale0_start_source=stage3.'
    vae_ckpt_blob = torch.load(args.vae_ckpt, map_location='cpu')
    ckpt_img_channels = int(_read_ckpt_arg(vae_ckpt_blob, 'img_channels', args.img_channels))
    if ckpt_img_channels != int(args.img_channels):
        print(
            f'[img_channels] overriding args.img_channels={args.img_channels} '
            f'to match vae_ckpt args.img_channels={ckpt_img_channels}'
        )
        args.img_channels = ckpt_img_channels
    args.img_channels = int(args.img_channels)
    if args.img_channels not in (1, 3):
        raise ValueError(f'img_channels must be 1 or 3, got {args.img_channels}')
    
    # =============== build dataset ===============
    print(f'[build PT data] ...\n')
    # Validate LR-condition switch matrix early; see SKILL: var-lr-data-conventions.
    if args.lr_cond_source == 'lr_vae':
        assert args.lr_folder in ('LR_64x64',), (
            f'lr_cond_source=lr_vae requires lr_folder=LR_64x64 (LR_VAE expects 64x64 input); '
            f'got lr_folder={args.lr_folder!r}'
        )
        assert args.stage1_ckpt, (
            f'lr_cond_source=lr_vae requires --stage1_ckpt to be non-empty.'
        )
    dataset_train, dataset_val = build_dataset(
        args.data_path, augment=True, use_ref=args.use_ref,
        lr_folder=args.lr_folder, hr_folder=args.hr_folder, same_shape=args.same_shape,
        img_channels=args.img_channels,
    )
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))
    
    # QUESTION: don't know why the batch_size is 1.5 times the args.batch_size
    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val
    
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        ),
    )
    del dataset_train
    
    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    # =============== build model ===============
    vae_local, srvar_wo_ddp = build_vae_srvar(
        args,
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        V=args.vocab_size, Cvae=args.Ct5, ch=args.vae_ch,
        share_quant_resi=args.share_quant_resi,
    )

    vae_ckpt = vae_ckpt_blob
    if isinstance(vae_ckpt, dict) and "trainer" in vae_ckpt.keys():
        # Try in priority order: vae_ema -> vae_wo_ddp -> vae -> first vae-like key.
        trainer_blob = vae_ckpt["trainer"]
        for key in ("vae_ema", "vae_wo_ddp", "vae"):
            if key in trainer_blob:
                vae_ckpt = trainer_blob[key]
                print(f"[vae_ckpt] using trainer['{key}']")
                break
        else:
            raise KeyError(f"vae_ckpt['trainer'] has no vae_ema/vae_wo_ddp/vae; keys={list(trainer_blob)}")
    vae_local.load_state_dict(vae_ckpt, strict=True)
    print(f"loaded vae from {args.vae_ckpt}")

    # Sanity: patch_nums in ckpt and args must agree (already encoded in tensor shapes
    # of quant_resi/mean_logvar_conv, so strict=True above would have failed otherwise).
    assert tuple(vae_local.quantize.v_patch_nums) == tuple(args.patch_nums), (
        f'patch_nums mismatch: ckpt={vae_local.quantize.v_patch_nums} vs args={args.patch_nums}'
    )

    stage3_encoder = _build_stage3_encoder(args, vae_local)

    # Optional stage1 LR_VAE.
    lr_vae_local: Optional[LR_VAE] = None
    if args.stage1_ckpt:
        lr_vae_local = build_lr_vae(args, dist.get_device(), Cvae=args.Ct5, ch=args.vae_ch)
        lr_ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
        if isinstance(lr_ckpt, dict) and 'trainer' in lr_ckpt:
            blob = lr_ckpt['trainer']
            for key in ('lr_vae_ema', 'lr_vae_wo_ddp', 'lr_vae'):
                if key in blob:
                    lr_ckpt = blob[key]
                    print(f"[stage1_ckpt] using trainer['{key}']")
                    break
            else:
                raise KeyError(f"stage1_ckpt['trainer'] has no lr_vae_ema/lr_vae_wo_ddp/lr_vae; keys={list(blob)}")
        lr_vae_local.load_state_dict(lr_ckpt, strict=False)
        print(f"[stage1_ckpt] loaded LR_VAE from {args.stage1_ckpt}")

        # Validate scale[0] dimension agreement.
        if dist.is_master():
            with torch.no_grad():
                dummy = torch.zeros(1, args.img_channels, 64, 64, device=dist.get_device())
                lr_mean = lr_vae_local.encode_to_posterior_mean(dummy)
                pn0 = args.patch_nums[0]
                assert lr_mean.shape[-1] == pn0, (
                    f'patch_nums[0]={pn0} must equal LR_VAE latent edge {lr_mean.shape[-1]} '
                    f'when stage1_ckpt is enabled.'
                )

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)

    if args.tini < 0:
        args.tini = math.sqrt(1 / srvar_wo_ddp.C / 3)
    srvar_wo_ddp.init_weights(other_std=args.tini)
    srvar_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)
    srvar_wo_ddp.init_LREncoder(vae_local)

    print(f'[PT] srvar model = {srvar_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('srvar', srvar_wo_ddp),
    )]) + '\n\n')
    
    srvar_wo_ddp = args.compile_model(srvar_wo_ddp, args.tfast)
    srvar_ddp_ema = None
    ddp_class = DDP if dist.initialized() else NullDDP # zero等于0的分支
    dist.barrier() # ADDED：wait for all processes to finish initialization, see the init_sync in DDP
    srvar_ddp: DDP = ddp_class(srvar_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    ndim_dict = {name: para.ndim for name, para in srvar_wo_ddp.named_parameters() if para.requires_grad}
    
    nowd_keys = set()
    nowd_keys |= {
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        'text_proj_for_sos.ca.mat_q',
    }
    names, paras, para_groups = filter_params(srvar_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if not args.ada:
        beta0, beta1 = 0.9, 0.999  # AdamW defaults when --ada is omitted
    elif '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1

    # build optimizer
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    srvar_optim = AmpOptimizer('srvar', args.fp16, opt_clz(params=para_groups, **opt_kw), srvar_wo_ddp, args.r_accu, args.tclip, args.zero)
    del names, paras, para_groups
    
    # build trainer
    trainer = SRVARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, srvar_wo_ddp=srvar_wo_ddp, srvar=srvar_ddp,
        var_opt=srvar_optim, label_smooth=args.ls,
        use_are_loss_weight=args.use_are_loss_weight,
        lr_vae=lr_vae_local,
        stage3_encoder=stage3_encoder,
        lr_cond_source=args.lr_cond_source,
        skip_scale0_loss=args.skip_scale0_loss,
        scale0_start_source=args.scale0_start_source,
        stage3_context_mode=args.stage3_context_mode,
        diffloss_batch_mul=args.diffloss_batch_mul,
        args=args,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)

    del vae_local, srvar_wo_ddp, srvar_ddp, srvar_optim, lr_vae_local, stage3_encoder
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_train_loss = 1e9
    best_val_loss = 1e9
    best_val_psnr = -1.0
    best_val_ssim = -1.0

    train_loss = -1.0
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)

        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )

        train_loss = stats.get('Ld', -1.0)
        grad_norm = stats.get('tnm', -1.0)
        best_train_loss = min(best_train_loss, train_loss)
        args.L_mean, args.grad_norm = train_loss, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        AR_ep_loss = dict(train_diff_loss=train_loss)
        is_val_and_also_saving = (ep + 1) % args.val_and_saving_per_ep == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_diff_loss, val_psnr, val_ssim, tot, cost = trainer.eval_ep(
                ld_val, use_ref=args.use_ref,
                eval_ar_max_batches=args.eval_ar_max_batches,
                cfg_infer_scale=args.cfg_infer,
            )
            best_updated = val_diff_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_diff_loss)
            best_val_psnr = max(best_val_psnr, val_psnr)
            best_val_ssim = max(best_val_ssim, val_ssim)
            AR_ep_loss.update(val_diff_loss=val_diff_loss, val_psnr=val_psnr, val_ssim=val_ssim)
            args.vL_mean, args.vacc_mean = val_diff_loss, val_psnr
            print(
                f' [*] [ep{ep}]  (val {tot})  '
                f'val_diff_loss: {val_diff_loss:.4f}, val_PSNR: {val_psnr:.2f}, val_SSIM: {val_ssim:.4f}  '
                f'Val cost: {cost:.2f}s'
            )

            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                    
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'ckpt-{ep+1}.pth')        
                torch.save({
                'epoch':    ep+1,
                'iter':     0,
                'trainer':  trainer.state_dict(),
                'args':     args.state_dict(),
                }, local_out_ckpt)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
                    
            dist.barrier()
        
        print(
            f'     [ep{ep}]  (training)  diff_loss: {best_train_loss:.4f} ({train_loss:.4f}),  '
            f'best_val_PSNR: {best_val_psnr:.2f},  best_val_SSIM: {best_val_ssim:.4f},  '
            f'Remain: {remain_time},  Finish: {finish_time}',
            flush=True,
        )
        tb_lg.update(head='AR_ep_loss', step=ep + 1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep + 1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log()
        tb_lg.flush()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   '
          f'best diff_loss: {best_train_loss:.4f} (last {train_loss:.4f}),   '
          f'best val PSNR/SSIM: {best_val_psnr:.2f}/{best_val_ssim:.4f}')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer:SRVARTrainer):
    # import heavy packages after Dataloader object creation
    from utils.lr_control import lr_wd_annealing

    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    me.add_meter('opt_step', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    me.add_meter('Ld', misc.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    log_points = max(1, int(getattr(args, 'train_log_points_per_epoch', 32)))
    for it, datas in me.log_every(start_it, iters_train, ld_or_itrt, log_points, header):
        if args.use_ref:
            low, super, ref = datas
        else :
            low, super = datas
            ref = None
        g_it = ep * iters_train + it
        if it < start_it: continue
        # if is_first_ep and it == start_it: warnings.resetwarnings()
        
        low = low.to(args.device, non_blocking=True)
        super = super.to(args.device, non_blocking=True)
        ref = ref.to(args.device, non_blocking=True) if ref is not None else None
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        progress = g_it / (max_it - 1)
        clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
        
        grad_norm, scale_log2 = trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping,clip_decay_ratio=clip_decay_ratio, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW_low=low, inp_B3HW_super=super, ref_B3HW = ref , prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        me.update(tlr=max_tlr, opt_step=(g_it + 1) // args.ac)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
