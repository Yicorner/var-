"""SRVARTrainer: continuous AR trainer for var/.

Replaces the discrete CrossEntropy-based trainer. The main loss is DiffLoss; we
keep auxiliary metrics (latent_mse, train_psnr/train_ssim, val PSNR/SSIM) for
visibility during training.
"""
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import SRVAR, VQVAE
from utils.amp_sc import AmpOptimizer
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor


_scale_schedule_fallback_warned = False


def _resolve_scale_schedule(
    inp_low: torch.Tensor,
    patch_nums: Tuple[int, ...],
    pn_str: str = "1M",
) -> List[Tuple[int, int, int]]:
    """Pick the (t, h, w) scale schedule.

    Prefers `dynamic_resolution_h_w[h_div_w]['1M']['scales']` for compat with
    the original codebase; falls back to `[(1, pn, pn) for pn in patch_nums]`
    when the lengths disagree.
    """
    h_div_w = inp_low.shape[-2] / inp_low.shape[-1]
    T = 1 if inp_low.dim() == 4 else inp_low.shape[2]
    keys = np.array(list(dynamic_resolution_h_w.keys()))
    template = keys[np.argmin(np.abs(h_div_w - keys))]
    scales = dynamic_resolution_h_w[template][pn_str]['scales']
    scales = [(min(t, T // 4 + 1), h, w) for (t, h, w) in scales]
    if len(scales) != len(patch_nums):
        scales = [(1, pn, pn) for pn in patch_nums]
        global _scale_schedule_fallback_warned
        if dist.is_master() and not _scale_schedule_fallback_warned:
            print(f'[scale_schedule] fallback to [(1,pn,pn)]; patch_nums={patch_nums}')
            _scale_schedule_fallback_warned = True
    else:
        for i, (t, h, w) in enumerate(scales):
            assert h == patch_nums[i] and w == patch_nums[i], \
                f'scale_schedule[{i}]=({t},{h},{w}) inconsistent with patch_nums[{i}]={patch_nums[i]}'
    return scales


class SRVARTrainer(object):
    def __init__(
        self,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        vae_local: VQVAE,
        srvar_wo_ddp: SRVAR,
        srvar: DDP,
        var_opt: AmpOptimizer,
        label_smooth: float = 0.0,                  # kept for ckpt compat, unused (no CE)
        use_are_loss_weight: bool = False,          # kept for ckpt compat, unused
        lr_vae=None,                                # frozen LR_VAE (optional, plan-A)
        lr_cond_source: str = 'srvar_encoder',
        skip_scale0_loss: bool = False,
        diffloss_batch_mul: int = 4,
        args=None,                                  # full args (for reconstruction metadata)
    ):
        super().__init__()
        self.srvar, self.vae_local = srvar, vae_local
        self.quantize_local = vae_local.quantize
        self.srvar_wo_ddp: SRVAR = srvar_wo_ddp
        self.var_opt = var_opt
        self.lr_vae = lr_vae
        self.lr_cond_source = lr_cond_source
        self.skip_scale0_loss = skip_scale0_loss
        self.diffloss_batch_mul = int(diffloss_batch_mul)
        self.args = args

        del self.srvar_wo_ddp.rng
        self.srvar_wo_ddp.rng = torch.Generator(device=device)

        self.patch_nums, self.resos = patch_nums, resos
        self.L = sum(pn * pn for pn in patch_nums)
        self.label_smooth = label_smooth
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        # Tracks whether we've already written the `run_metadata.json` for this run.
        self._reconstruction_metadata_written: bool = False
        # Resolved once per run (LR spatial size is fixed for paired SR training).
        self._cached_scale_schedule: Optional[List[Tuple[int, int, int]]] = None

    def _get_scale_schedule(self, inp_B3HW_low: Ten) -> List[Tuple[int, int, int]]:
        if self._cached_scale_schedule is None:
            self._cached_scale_schedule = _resolve_scale_schedule(inp_B3HW_low, self.patch_nums)
        return self._cached_scale_schedule

    # ----------------------------------------------------------------- helpers
    def _build_targets(self, inp_B3HW_super: Ten):
        """Run the frozen VAE to get `(ms_h_target, ms_x_input)` under no_grad."""
        with torch.no_grad():
            ms_h_target, ms_x_input, f_hat_full = self.vae_local.img_to_ms_continuous_input(inp_B3HW_super)
        return ms_h_target, ms_x_input, f_hat_full

    @staticmethod
    def _tensor_stats(name: str, tensor: Optional[Ten]) -> str:
        if tensor is None:
            return f'{name}=None'
        t = tensor.detach().float()
        return (
            f'{name}: shape={tuple(t.shape)} '
            f'mean={t.mean().item():+.4f} std={t.std(unbiased=False).item():.4f} '
            f'min={t.min().item():+.4f} max={t.max().item():+.4f}'
        )

    @staticmethod
    def _should_run_event(it: int, metric_lg: MetricLogger, interval: int) -> bool:
        return (
            it == 0
            or it in metric_lg.log_iters
            or (interval > 0 and it % interval == 0)
        )

    def _maybe_lr_vae_override(
        self,
        inp_B3HW_low: Ten,
        ms_h_target: List[Ten],
    ) -> Tuple[List[Ten], Optional[Ten]]:
        """When stage1 LR_VAE is enabled, override `ms_h_target[0]` with the
        LR_VAE encoded scale[0] latent. Also return the LR_VAE-derived `low_f`
        when `lr_cond_source='lr_vae'`."""
        low_f_override: Optional[Ten] = None
        if self.lr_vae is not None:
            with torch.no_grad():
                lr_mean = self.lr_vae.encode_to_posterior_mean(inp_B3HW_low)  # [B, C, h0, w0]
            # Sanity on scale[0] shape consistency.
            B, C, h, w = lr_mean.shape
            assert ms_h_target[0].shape == lr_mean.shape, (
                f'LR_VAE latent {tuple(lr_mean.shape)} != scale[0] target '
                f'{tuple(ms_h_target[0].shape)}; check patch_nums[0] vs lr_img_size/16'
            )
            ms_h_target[0] = lr_mean.contiguous()
            if self.lr_cond_source == 'lr_vae':
                low_f_override = lr_mean.reshape(B, C, -1).transpose(1, 2).contiguous()  # [B, h*w, C]
        return ms_h_target, low_f_override

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def eval_ep(
        self,
        ld_val: DataLoader,
        use_ref: bool,
        eval_ar_max_batches: int = 4,
        cfg_infer_scale: float = 1.0,
        temperature: float = 1.0,
    ):
        """Compute val DiffLoss; additionally run AR inference on up to
        `eval_ar_max_batches` batches and compute PSNR/SSIM.
        """
        tot = 0
        diff_loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        ar_count = 0
        stt = time.time()
        training = self.srvar_wo_ddp.training
        self.srvar_wo_ddp.eval()

        for batch_idx, datas in enumerate(ld_val):
            if use_ref:
                inp_B3HW_low, inp_B3HW_super, ref_B3HW = datas
                ref_B3HW = ref_B3HW.to(dist.get_device(), non_blocking=True)
            else:
                inp_B3HW_low, inp_B3HW_super = datas
                ref_B3HW = None
            inp_B3HW_low = inp_B3HW_low.to(dist.get_device(), non_blocking=True)
            inp_B3HW_super = inp_B3HW_super.to(dist.get_device(), non_blocking=True)
            B = inp_B3HW_low.shape[0]

            ms_h_target, ms_x_input, _ = self._build_targets(inp_B3HW_super)
            ms_h_target, low_f_override = self._maybe_lr_vae_override(inp_B3HW_low, ms_h_target)
            scale_schedule = self._get_scale_schedule(inp_B3HW_low)

            loss = self.srvar(
                inp_B3HW_low=inp_B3HW_low,
                ms_h_target=ms_h_target,
                ms_x_input=ms_x_input,
                scale_schedule=scale_schedule,
                ref_B3HW=ref_B3HW,
                low_f_override=low_f_override,
                cfg_infer=True,
                scale0_loss_mask=self.skip_scale0_loss,
            )
            diff_loss_sum += float(loss.item()) * B
            tot += B

            # AR-based PSNR/SSIM on a few batches.
            if ar_count < eval_ar_max_batches:
                try:
                    from utils.image_saver import compute_psnr_ssim
                    rec_img = self._quick_reconstruction(
                        inp_B3HW_low, scale_schedule, ref_B3HW, low_f_override,
                        cfg=cfg_infer_scale, temperature=temperature,
                    )
                    metrics = compute_psnr_ssim(rec_img, inp_B3HW_super)
                    psnr_sum += metrics['psnr_mean'] * B
                    ssim_sum += metrics['ssim_mean'] * B
                    ar_count += 1
                except Exception as e:
                    if dist.is_master():
                        print(f'[eval_ep] AR inference failed on batch {batch_idx}: {e}')

        self.srvar_wo_ddp.train(training)

        stats = torch.tensor(
            [diff_loss_sum, psnr_sum, ssim_sum, float(tot), float(ar_count)],
            device=dist.get_device(),
        )
        dist.allreduce(stats)
        tot = int(round(stats[3].item())) or 1
        ar_count_total = int(round(stats[4].item())) or 1
        diff_loss = float(stats[0].item() / tot)
        # AR metrics are averaged over batches that actually ran AR.
        psnr_mean = float(stats[1].item() / max(1, tot)) if ar_count > 0 else 0.0
        ssim_mean = float(stats[2].item() / max(1, tot)) if ar_count > 0 else 0.0
        return diff_loss, psnr_mean, ssim_mean, tot, time.time() - stt

    @torch.no_grad()
    def _sample_ar(
        self,
        inp_B3HW_low: Ten,
        scale_schedule: List[Tuple[int, int, int]],
        ref_B3HW: Optional[Ten],
        low_f_override: Optional[Ten],
        cfg: float = 1.0,
        temperature: float = 1.0,
        trunk_scale: int = 1000,
    ) -> Tuple[Ten, List[Ten], Ten]:
        """Run AR sampling and return `(rec_img, sampled_tokens, f_hat)`."""
        was_training = self.srvar_wo_ddp.training
        self.srvar_wo_ddp.eval()
        try:
            sampled_tokens, fhat_tup, _ = self.srvar_wo_ddp.autoregressive_infer_cfg(
                vae=self.vae_local,
                scale_schedule=scale_schedule,
                inp_B3HW_low=inp_B3HW_low,
                ref_B3HW=ref_B3HW,
                low_f_override=low_f_override,
                B=inp_B3HW_low.shape[0],
                return_fhat=True,
                cfg=cfg,
                temperature=temperature,
                trunk_scale=trunk_scale,
            )
            f_hat = fhat_tup[0]
            rec = self.vae_local.fhat_to_img(f_hat)
        finally:
            self.srvar_wo_ddp.train(was_training)
        return rec, sampled_tokens, f_hat

    @torch.no_grad()
    def _decode_target_scale0_only(self, target_s0: Ten) -> Ten:
        """Decode the true first-scale VAE target alone for diagnostic ceiling."""
        B, C, _, _ = target_s0.shape
        final_pn = int(self.patch_nums[-1])
        accu = target_s0.new_zeros(B, C, final_pn, final_pn)
        accu, _ = self.vae_local.quantize.get_next_autoregressive_input(
            0, len(self.patch_nums), accu, target_s0,
        )
        return self.vae_local.fhat_to_img(accu)

    @torch.no_grad()
    def _quick_reconstruction(
        self,
        inp_B3HW_low: Ten,
        scale_schedule: List[Tuple[int, int, int]],
        ref_B3HW: Optional[Ten],
        low_f_override: Optional[Ten],
        cfg: float = 1.0,
        temperature: float = 1.0,
    ) -> Ten:
        """Run AR sampling and return reconstructed HR `[B, 3, H, W]` in [-1, 1]."""
        rec, _, _ = self._sample_ar(
            inp_B3HW_low, scale_schedule, ref_B3HW, low_f_override,
            cfg=cfg, temperature=temperature,
        )
        return rec

    # ----------------------------------------------------------------- train
    def train_step(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        clip_decay_ratio: float,
        metric_lg: MetricLogger,
        tb_lg: TensorboardLogger,
        inp_B3HW_low: Ten,
        inp_B3HW_super: Ten,
        ref_B3HW: Optional[Ten],
        prog_si: int,
        prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """One training step on a (LR, HR[, ref]) batch."""
        # Progressive training compat (passed through; quant.prog_si stays -1).
        self.srvar_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1:
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1

        self.srvar.require_backward_grad_sync = stepping

        B = inp_B3HW_low.shape[0]
        ms_h_target, ms_x_input, f_hat_full = self._build_targets(inp_B3HW_super)
        ms_h_target, low_f_override = self._maybe_lr_vae_override(inp_B3HW_low, ms_h_target)
        scale_schedule = self._get_scale_schedule(inp_B3HW_low)

        with self.var_opt.amp_ctx:
            loss = self.srvar(
                inp_B3HW_low=inp_B3HW_low,
                ms_h_target=ms_h_target,
                ms_x_input=ms_x_input,
                scale_schedule=scale_schedule,
                ref_B3HW=ref_B3HW,
                low_f_override=low_f_override,
                cfg_infer=False,
                scale0_loss_mask=self.skip_scale0_loss,
            )

        grad_norm, scale_log2 = self.var_opt.backward_clip_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, loss=loss, clip_decay_ratio=clip_decay_ratio,
        )

        recon_interval = int(getattr(self.args, 'reconstruction_save_interval', 0) or 0) if self.args is not None else 0
        diag_interval = int(getattr(self.args, 'diagnostics_interval', 0) or 0) if self.args is not None else 0
        log_event = self._should_run_event(it, metric_lg, interval=0)
        recon_event = self._should_run_event(it, metric_lg, interval=recon_interval)
        diag_event = (
            self.args is not None
            and getattr(self.args, 'diagnostics_enabled', True)
            and self._should_run_event(it, metric_lg, interval=diag_interval)
        )

        # Light-weight metric logging on the configured log iterations.
        if log_event:
            metric_kwargs = dict(Ld=float(loss.item()), step=g_it)
            if grad_norm is not None:
                metric_kwargs['tnm'] = (
                    grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                )
            metric_lg.update(**metric_kwargs)

            # Reconstruction visualization + auxiliary PSNR (cheap-ish).
        if self.args is not None and (recon_event or diag_event):
            if getattr(self.args, 'save_reconstruction_images', True) or diag_event:
                try:
                    self._maybe_save_reconstruction(
                        ep, it, inp_B3HW_low, inp_B3HW_super, ref_B3HW,
                        low_f_override, scale_schedule, f_hat_full, ms_h_target,
                        save_reconstruction=bool(getattr(self.args, 'save_reconstruction_images', True) and recon_event),
                        save_diagnostics=bool(diag_event),
                    )
                except Exception as e:
                    if dist.is_master():
                        print(f'[train_step] reconstruction/diagnostics skipped: {e}')

        if g_it == 0 or (g_it + 1) % 500 == 0:
            if dist.is_master():
                tb_lg.update(
                    head='AR_iter_loss',
                    diff_loss=float(loss.item()),
                    step=g_it,
                )
                tb_lg.update(head='AR_iter_schedule', prog_si=prog_si, prog_wp=prog_wp, step=g_it)

        self.srvar_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    # ----------------------------------------------------------------- visualization
    def _maybe_save_reconstruction(
        self,
        ep: int,
        it: int,
        inp_B3HW_low: Ten,
        inp_B3HW_super: Ten,
        ref_B3HW: Optional[Ten],
        low_f_override: Optional[Ten],
        scale_schedule: List[Tuple[int, int, int]],
        f_hat_full: Ten,
        ms_h_target: List[Ten],
        save_reconstruction: bool = True,
        save_diagnostics: bool = False,
    ):
        """Render reconstruction and diagnostic grids on selected iterations."""
        if not dist.is_master():
            return
        from utils.image_saver import save_reconstruction_comparison
        from utils.image_saver import save_reconstruction_run_metadata
        from utils.image_saver import save_diagnostic_comparison
        from utils.image_saver import compute_psnr_ssim

        args = self.args
        save_dir = os.path.join(
            getattr(args, 'local_out_dir_path', './local_output'),
            getattr(args, 'reconstruction_dir_name', 'reconstruction_samples'),
        )

        max_samples = int(getattr(args, 'reconstruction_max_samples', 4))
        max_B = min(inp_B3HW_low.shape[0], max_samples)
        with torch.no_grad():
            rec, sampled_tokens, _ = self._sample_ar(
                inp_B3HW_low[:max_B], scale_schedule,
                ref_B3HW[:max_B] if ref_B3HW is not None else None,
                low_f_override[:max_B] if low_f_override is not None else None,
            )
            oracle = self.vae_local.fhat_to_img(f_hat_full[:max_B])

        if save_reconstruction:
            save_reconstruction_comparison(
                lr=inp_B3HW_low[:max_B],
                hr_pred=rec,
                hr_gt=inp_B3HW_super[:max_B],
                save_dir=save_dir,
                ep=ep,
                it=it,
                max_samples=max_samples,
            )

        if save_diagnostics:
            diag_dir = os.path.join(
                getattr(args, 'local_out_dir_path', './local_output'),
                getattr(args, 'diagnostics_dir_name', 'diagnostics'),
            )
            rec_scale0 = None
            target_scale0_oracle = None
            scale0_tokens = []
            if getattr(args, 'diagnostics_sample_scale0', True):
                rec_scale0, scale0_tokens, _ = self._sample_ar(
                    inp_B3HW_low[:max_B], scale_schedule,
                    ref_B3HW[:max_B] if ref_B3HW is not None else None,
                    low_f_override[:max_B] if low_f_override is not None else None,
                    trunk_scale=1,
                )
                target_scale0_oracle = self._decode_target_scale0_only(ms_h_target[0][:max_B])
            save_diagnostic_comparison(
                lr=inp_B3HW_low[:max_B],
                hr_ar=rec,
                hr_scale0=rec_scale0,
                hr_scale0_oracle=target_scale0_oracle,
                hr_oracle=oracle,
                hr_gt=inp_B3HW_super[:max_B],
                save_dir=diag_dir,
                ep=ep,
                it=it,
                max_samples=int(getattr(args, 'diagnostics_max_samples', max_samples)),
            )

            ar_metrics = compute_psnr_ssim(rec, inp_B3HW_super[:max_B])
            oracle_metrics = compute_psnr_ssim(oracle, inp_B3HW_super[:max_B])
            scale0_metrics = compute_psnr_ssim(rec_scale0, inp_B3HW_super[:max_B]) if rec_scale0 is not None else None
            target_scale0_metrics = compute_psnr_ssim(target_scale0_oracle, inp_B3HW_super[:max_B]) if target_scale0_oracle is not None else None
            print(
                f'[diagnostics ep={ep} it={it}] '
                f'AR_PSNR={ar_metrics["psnr_mean"]:.2f} AR_SSIM={ar_metrics["ssim_mean"]:.4f} | '
                f'ORACLE_PSNR={oracle_metrics["psnr_mean"]:.2f} ORACLE_SSIM={oracle_metrics["ssim_mean"]:.4f}'
                + (
                    f' | SCALE0_PSNR={scale0_metrics["psnr_mean"]:.2f} SCALE0_SSIM={scale0_metrics["ssim_mean"]:.4f}'
                    if scale0_metrics is not None else ''
                )
                + (
                    f' | TARGET_SCALE0_PSNR={target_scale0_metrics["psnr_mean"]:.2f} TARGET_SCALE0_SSIM={target_scale0_metrics["ssim_mean"]:.4f}'
                    if target_scale0_metrics is not None else ''
                )
            )
            print('[diagnostics latent] ' + ' | '.join([
                self._tensor_stats('lr', inp_B3HW_low[:max_B]),
                self._tensor_stats('target_s0', ms_h_target[0][:max_B]),
                self._tensor_stats('target_last', ms_h_target[-1][:max_B]),
                self._tensor_stats('sample_s0', sampled_tokens[0] if sampled_tokens else None),
                self._tensor_stats('sample_last', sampled_tokens[-1] if sampled_tokens else None),
                self._tensor_stats('sample_s0_only', scale0_tokens[0] if scale0_tokens else None),
            ]))

        if not self._reconstruction_metadata_written and getattr(args, 'record_reconstruction_metadata', True):
            save_reconstruction_run_metadata(
                save_dir=save_dir,
                args_state=args.state_dict() if hasattr(args, 'state_dict') else {},
                stage_name="SRVAR continuous AR (DiffLoss head)",
                frequency_description=(
                    "Every train_log iter (and optionally every N iters via reconstruction_save_interval)"
                ),
                max_samples=max_samples,
            )
            self._reconstruction_metadata_written = True

    # ----------------------------------------------------------------- ckpt
    def get_config(self):
        return {
            'patch_nums': self.patch_nums,
            'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it': self.prog_it,
            'last_prog_si': self.last_prog_si,
            'first_prog': self.first_prog,
            'lr_cond_source': self.lr_cond_source,
            'skip_scale0_loss': self.skip_scale0_loss,
            'diffloss_batch_mul': self.diffloss_batch_mul,
            'scale0_query_source': getattr(self.srvar_wo_ddp, 'scale0_query_source', 'sos'),
            'scale_loss_weighting': getattr(self.srvar_wo_ddp, 'scale_loss_weighting', 'token'),
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('srvar_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        if self.lr_vae is not None:
            state['lr_vae'] = self.lr_vae.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('srvar_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k:
                print("load var and skip var's vaex!")
                continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        if self.lr_vae is not None and 'lr_vae' in state:
            self.lr_vae.load_state_dict(state['lr_vae'], strict=strict)

        config: dict = state.pop('config', None)
        if config is not None:
            self.prog_it = config.get('prog_it', 0)
            self.last_prog_si = config.get('last_prog_si', -1)
            self.first_prog = config.get('first_prog', True)
            for k, v in self.get_config().items():
                if k not in config:
                    continue
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
