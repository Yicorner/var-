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
from models import SRVAR, Stage3Scale0Encoder, VQVAE
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
        stage3_encoder: Optional[Stage3Scale0Encoder] = None,
        lr_cond_source: str = 'srvar_encoder',
        skip_scale0_loss: bool = False,
        scale0_start_source: str = 'transformer',
        stage3_context_mode: str = 'both',
        diffloss_batch_mul: int = 4,
        args=None,                                  # full args (for reconstruction metadata)
    ):
        super().__init__()
        self.srvar, self.vae_local = srvar, vae_local
        self.quantize_local = vae_local.quantize
        self.srvar_wo_ddp: SRVAR = srvar_wo_ddp
        self.var_opt = var_opt
        self.lr_vae = lr_vae
        self.stage3_encoder = stage3_encoder
        self.lr_cond_source = lr_cond_source
        self.scale0_start_source = scale0_start_source
        self.stage3_context_mode = stage3_context_mode
        self.skip_scale0_loss = skip_scale0_loss
        self.diffloss_batch_mul = int(diffloss_batch_mul)
        self.args = args

        assert self.scale0_start_source in ('transformer', 'stage3')
        assert self.stage3_context_mode in ('both', 'prefix_only')
        assert self.lr_cond_source in ('srvar_encoder', 'learned_lr_encoder', 'lr_vae')
        if self.scale0_start_source == 'stage3':
            assert self.stage3_encoder is not None, 'scale0_start_source=stage3 requires stage3_encoder.'
        if self.stage3_encoder is not None:
            self.stage3_encoder.eval()
            for p in self.stage3_encoder.parameters():
                p.requires_grad_(False)

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

    def _uses_stage3_start(self) -> bool:
        return self.scale0_start_source == 'stage3'

    @torch.no_grad()
    def _stage3_s0(self, inp_B3HW_low: Ten) -> Optional[Ten]:
        """Frozen myvaex stage3 LR -> scale[0] posterior mean."""
        if not self._uses_stage3_start():
            return None
        assert self.stage3_encoder is not None
        was_training = self.stage3_encoder.training
        self.stage3_encoder.eval()
        try:
            return self.stage3_encoder(inp_B3HW_low).contiguous()
        finally:
            self.stage3_encoder.train(was_training)

    @torch.no_grad()
    def _replace_s1_input_with_stage3_s0(
        self,
        ms_x_input: Optional[Ten],
        stage3_s0: Optional[Ten],
        scale_schedule: List[Tuple[int, int, int]],
    ) -> Optional[Ten]:
        """Use stage3 s0 only for the first teacher-forcing transition.

        `ms_x_input` layout is `[input_for_s1, input_for_s2, ...]`; this method
        replaces only the first segment. Inputs for s2+ remain HR teacher-forced.
        """
        if stage3_s0 is None:
            return ms_x_input
        if len(scale_schedule) <= 1:
            assert ms_x_input is None, 'single-scale schedule should not have ms_x_input.'
            return None
        assert ms_x_input is not None, 'stage3 schedule with s1+ requires ms_x_input.'
        pn_t, pn_h, pn_w = scale_schedule[0]
        assert pn_t == 1 and stage3_s0.shape[-2:] == (pn_h, pn_w), (
            f'stage3_s0 {tuple(stage3_s0.shape)} incompatible with scale[0]={scale_schedule[0]}'
        )
        next_t, next_h, next_w = scale_schedule[1]
        assert next_t == 1, f'stage3 s1 input replacement expects image scale[1], got {scale_schedule[1]}'

        B, C, _, _ = stage3_s0.shape
        final_pn = int(self.patch_nums[-1])
        accu = stage3_s0.new_zeros(B, C, final_pn, final_pn)
        _, s1_fhat = self.vae_local.quantize.get_next_autoregressive_input(
            0, len(self.patch_nums), accu, stage3_s0,
        )
        s1_tokens = s1_fhat.reshape(B, C, -1).transpose(1, 2).contiguous()
        first_next_l = int(next_t * next_h * next_w)
        assert s1_tokens.shape[1] == first_next_l, (
            f'stage3-derived s1 tokens {s1_tokens.shape[1]} != expected {first_next_l}'
        )
        assert ms_x_input.shape[1] >= first_next_l, (
            f'ms_x_input length {ms_x_input.shape[1]} shorter than s1 segment {first_next_l}'
        )
        return torch.cat([s1_tokens, ms_x_input[:, first_next_l:]], dim=1)

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
    def _format_per_scale_latent_stats(stats) -> str:
        if not stats:
            return ''
        parts = []
        for idx, item in enumerate(stats):
            si = int(item.get('scale', idx))
            h = int(item.get('h', 0))
            w = int(item.get('w', 0))
            n = int(item.get('tokens', 0))
            mse = float(item.get('mse', float('nan')))
            psnr = float(item.get('psnr', float('nan')))
            parts.append(f's{si}({h}x{w},n={n}):mse={mse:.4e},psnr={psnr:.2f}dB')
        return ' | '.join(parts)

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
            if not self._uses_stage3_start():
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
        ar_batches = 0
        ar_samples = 0
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
            stage3_s0 = self._stage3_s0(inp_B3HW_low)
            if stage3_s0 is not None:
                assert stage3_s0.shape == ms_h_target[0].shape, (
                    f'stage3_s0 {tuple(stage3_s0.shape)} != target_s0 {tuple(ms_h_target[0].shape)}'
                )
                ms_x_input = self._replace_s1_input_with_stage3_s0(
                    ms_x_input, stage3_s0, scale_schedule
                )

            loss = self.srvar(
                inp_B3HW_low=inp_B3HW_low,
                ms_h_target=ms_h_target,
                ms_x_input=ms_x_input,
                scale_schedule=scale_schedule,
                ref_B3HW=ref_B3HW,
                low_f_override=low_f_override,
                stage3_s0=stage3_s0,
                cfg_infer=True,
                scale0_loss_mask=self.skip_scale0_loss or self._uses_stage3_start(),
            )
            diff_loss_sum += float(loss.item()) * B
            tot += B

            # AR-based PSNR/SSIM on a few batches.
            if ar_batches < eval_ar_max_batches:
                try:
                    from utils.image_saver import compute_psnr_ssim
                    rec_img = self._quick_reconstruction(
                        inp_B3HW_low, scale_schedule, ref_B3HW, low_f_override,
                        stage3_s0=stage3_s0,
                        cfg=cfg_infer_scale, temperature=temperature,
                    )
                    metrics = compute_psnr_ssim(rec_img, inp_B3HW_super)
                    psnr_sum += metrics['psnr_mean'] * B
                    ssim_sum += metrics['ssim_mean'] * B
                    ar_batches += 1
                    ar_samples += B
                except Exception as e:
                    if dist.is_master():
                        print(f'[eval_ep] AR inference failed on batch {batch_idx}: {e}')

        self.srvar_wo_ddp.train(training)

        stats = torch.tensor(
            [diff_loss_sum, psnr_sum, ssim_sum, float(tot), float(ar_samples)],
            device=dist.get_device(),
        )
        dist.allreduce(stats)
        tot = int(round(stats[3].item())) or 1
        ar_samples_total = int(round(stats[4].item()))
        diff_loss = float(stats[0].item() / tot)
        # AR metrics are averaged over batches that actually ran AR.
        psnr_mean = float(stats[1].item() / max(1, ar_samples_total)) if ar_samples_total > 0 else 0.0
        ssim_mean = float(stats[2].item() / max(1, ar_samples_total)) if ar_samples_total > 0 else 0.0
        return diff_loss, psnr_mean, ssim_mean, tot, time.time() - stt

    @torch.no_grad()
    def _sample_ar(
        self,
        inp_B3HW_low: Ten,
        scale_schedule: List[Tuple[int, int, int]],
        ref_B3HW: Optional[Ten],
        low_f_override: Optional[Ten],
        stage3_s0: Optional[Ten] = None,
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
                stage3_s0=stage3_s0,
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
    def _decode_tokens_by_scale(self, tokens: List[Ten]) -> List[Ten]:
        """Decode cumulative AR reconstructions after each sampled scale."""
        if not tokens:
            return []
        B, C, _, _ = tokens[0].shape
        final_pn = int(self.patch_nums[-1])
        accu = tokens[0].new_zeros(B, C, final_pn, final_pn)
        recs: List[Ten] = []
        for si, h in enumerate(tokens):
            accu, _ = self.vae_local.quantize.get_next_autoregressive_input(
                si, len(self.patch_nums), accu, h,
            )
            recs.append(self.vae_local.fhat_to_img(accu.clone()))
        return recs

    @torch.no_grad()
    def _quick_reconstruction(
        self,
        inp_B3HW_low: Ten,
        scale_schedule: List[Tuple[int, int, int]],
        ref_B3HW: Optional[Ten],
        low_f_override: Optional[Ten],
        stage3_s0: Optional[Ten] = None,
        cfg: float = 1.0,
        temperature: float = 1.0,
    ) -> Ten:
        """Run AR sampling and return reconstructed HR `[B, img_channels, H, W]` in [-1, 1]."""
        rec, _, _ = self._sample_ar(
            inp_B3HW_low, scale_schedule, ref_B3HW, low_f_override,
            stage3_s0=stage3_s0,
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
        stage3_s0 = self._stage3_s0(inp_B3HW_low)
        if stage3_s0 is not None:
            assert stage3_s0.shape == ms_h_target[0].shape, (
                f'stage3_s0 {tuple(stage3_s0.shape)} != target_s0 {tuple(ms_h_target[0].shape)}'
            )
            ms_x_input = self._replace_s1_input_with_stage3_s0(
                ms_x_input, stage3_s0, scale_schedule
            )

        with self.var_opt.amp_ctx:
            loss = self.srvar(
                inp_B3HW_low=inp_B3HW_low,
                ms_h_target=ms_h_target,
                ms_x_input=ms_x_input,
                scale_schedule=scale_schedule,
                ref_B3HW=ref_B3HW,
                low_f_override=low_f_override,
                stage3_s0=stage3_s0,
                cfg_infer=False,
                scale0_loss_mask=self.skip_scale0_loss or self._uses_stage3_start(),
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
                        low_f_override, stage3_s0, scale_schedule, f_hat_full, ms_h_target,
                        save_reconstruction=bool(getattr(self.args, 'save_reconstruction_images', True) and recon_event),
                        save_diagnostics=bool(diag_event),
                    )
                except Exception as e:
                    if dist.is_master():
                        print(f'[train_step] reconstruction/diagnostics skipped: {e}')
            # Master-only AR sampling can take minutes; other ranks must not start
            # the next iter (DDP forward) until rank 0 finishes diagnostics.
            if dist.initialized():
                dist.barrier()

        if (log_event or diag_event) and dist.is_master():
            per_scale_log = self._format_per_scale_latent_stats(
                getattr(self.srvar_wo_ddp, 'latest_per_scale_stats', None)
            )
            if per_scale_log:
                print(f'[train per-scale latent ep={ep} it={it}] {per_scale_log}', flush=True)

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
        stage3_s0: Optional[Ten],
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
        from utils.image_saver import save_multiscale_diagnostic_comparison
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
                stage3_s0=stage3_s0[:max_B] if stage3_s0 is not None else None,
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
                    stage3_s0=stage3_s0[:max_B] if stage3_s0 is not None else None,
                    trunk_scale=1,
                )
                target_scale0_oracle = self._decode_target_scale0_only(ms_h_target[0][:max_B])
            stage3_scale0_oracle = (
                self._decode_target_scale0_only(stage3_s0[:max_B])
                if stage3_s0 is not None else None
            )
            save_diagnostic_comparison(
                lr=inp_B3HW_low[:max_B],
                hr_ar=rec,
                hr_scale0=rec_scale0,
                hr_scale0_oracle=target_scale0_oracle,
                hr_stage3_scale0=stage3_scale0_oracle,
                hr_oracle=oracle,
                hr_gt=inp_B3HW_super[:max_B],
                save_dir=diag_dir,
                ep=ep,
                it=it,
                max_samples=int(getattr(args, 'diagnostics_max_samples', max_samples)),
            )
            if getattr(args, 'diagnostics_multiscale', True):
                multiscale_recs = self._decode_tokens_by_scale(sampled_tokens)
                save_multiscale_diagnostic_comparison(
                    lr=inp_B3HW_low[:max_B],
                    hr_by_scale=multiscale_recs,
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
            stage3_scale0_metrics = compute_psnr_ssim(stage3_scale0_oracle, inp_B3HW_super[:max_B]) if stage3_scale0_oracle is not None else None
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
                + (
                    f' | STAGE3_SCALE0_PSNR={stage3_scale0_metrics["psnr_mean"]:.2f} STAGE3_SCALE0_SSIM={stage3_scale0_metrics["ssim_mean"]:.4f}'
                    if stage3_scale0_metrics is not None else ''
                )
            )
            print('[diagnostics latent] ' + ' | '.join([
                self._tensor_stats('lr', inp_B3HW_low[:max_B]),
                self._tensor_stats('stage3_s0', stage3_s0[:max_B] if stage3_s0 is not None else None),
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
            'scale0_start_source': self.scale0_start_source,
            'stage3_context_mode': self.stage3_context_mode,
            'diffloss_batch_mul': self.diffloss_batch_mul,
            'scale0_query_source': getattr(self.srvar_wo_ddp, 'scale0_query_source', 'sos'),
            'scale_loss_weighting': getattr(self.srvar_wo_ddp, 'scale_loss_weighting', 'token'),
            'gpt_embed_dim': getattr(self.srvar_wo_ddp, 'C', None),
            'gpt_depth': getattr(self.srvar_wo_ddp, 'depth', None),
            'gpt_num_heads': getattr(self.srvar_wo_ddp, 'num_heads', None),
            'gpt_mlp_ratio': getattr(self.srvar_wo_ddp, 'mlp_ratio', None),
            'learned_lr_encoder_width': getattr(self.srvar_wo_ddp, 'learned_lr_encoder_width', None),
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
        if self.stage3_encoder is not None:
            state['stage3_encoder'] = self.stage3_encoder.state_dict()
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
        if self.stage3_encoder is not None and 'stage3_encoder' in state:
            self.stage3_encoder.load_state_dict(state['stage3_encoder'], strict=strict)

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
