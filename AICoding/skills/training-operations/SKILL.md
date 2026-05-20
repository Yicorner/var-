---
name: training-operations
description: Collect var/ training parameters, SRtrain.sh shell-vars, log iterations, reconstruction visualization, run_metadata.json, and PSNR/SSIM evaluation.
---

# Training Operations (var)

> 面向跑实验、调参、看日志、看重建图的人。把 var 训练运维相关的东西单独收拢。

---

## 1. 参数速查

参数定义文件：`utils/arg_util.py`。下面只列出与 myvaex 不同 / 新增的部分；通用的 `bed/exp_name/ep/lbs/...` 沿用 myvaex 习惯。

### 1.1 VAE 接口（必须与 stage2 ckpt 一致）

| 参数 | 默认 | 说明 |
|---|---|---|
| `vae_ckpt` | 必填 | myvaex stage2 多尺度连续 VAE checkpoint 路径 |
| `patch_nums` | (1,2,...,16) | 必须与 ckpt 训练时一致，否则 quant_resi ticks 偏移 |
| `Ct5` / `Cvae` | 32 | latent 通道数 |
| `vae_ch` | 128 | encoder/decoder 基础通道 |
| `quant_resi` | 0.5 | residual conv 比例 |
| `share_quant_resi` | 4 | residual conv 跨尺度共享方式 |

### 1.2 DiffLoss 头

| 参数 | 默认 | 说明 |
|---|---|---|
| `diffloss_w` | 1024 | `SimpleMLPAdaLN` 宽度 |
| `diffloss_d` | 3 | `SimpleMLPAdaLN` 深度 |
| `diff_steps` | "100" | 推理用 spaced diffusion 步数（字符串） |
| `diffloss_batch_mul` | 4 | DiffLoss 内 N 维 repeat 倍数 |
| `scale_loss_weighting` | `token` | `token` 或 `equal_scale`；后者让各 scale 平均贡献接近一致 |
| `scale0_query_source` | `sos` | `sos` 或 `low_f_pool`；后者用 LR token 池化成 scale[0] query |

### 1.3 LR 路径

| 参数 | 默认 | 说明 |
|---|---|---|
| `lr_folder` | `LR_64x64` | LR 子目录名 |
| `hr_folder` | `HR` | HR 子目录名 |
| `lr_cond_source` | `srvar_encoder` | `srvar_encoder` 或 `lr_vae` |
| `stage1_ckpt` | `""` | 非空时构造 LR_VAE 并加载 |
| `skip_scale0_loss` | False | 启用 stage1 时是否跳过 scale[0] 的 DiffLoss |
| `tlen` | 1024 | `cfg_uncond` 长度，需 `≥ low_len` |

### 1.4 CFG

| 参数 | 默认 | 说明 |
|---|---|---|
| `cond_drop_rate` | 0.1 | 训练 CFG dropout |
| `cfg_infer` | 1.0 | 推理 CFG scale |

### 1.5 重建可视化与评估

| 参数 | 默认 | 说明 |
|---|---|---|
| `save_reconstruction_images` | True | 是否在 train log iter 保存重建图 |
| `reconstruction_save_interval` | 0 | 0 = 跟随 log iter；>0 = 每 N 个 iter 保存 |
| `reconstruction_max_samples` | 4 | 每张对比图最多多少行 |
| `reconstruction_dir_name` | `reconstruction_samples` | 输出子目录名 |
| `record_reconstruction_metadata` | True | 是否写 `run_metadata.json` |
| `val_and_saving_per_ep` | 2 | 每 N 个 epoch 验证并保存 ckpt |
| `eval_ar_max_batches` | 4 | eval_ep 中跑 AR 推理 PSNR/SSIM 的 batch 上限 |
| `diagnostics_enabled` | True | 是否保存诊断图并打印 latent/sample 统计 |
| `diagnostics_interval` | 0 | 0 = 跟随 log iter；>0 = 每 N iter 诊断一次 |
| `diagnostics_dir_name` | `diagnostics` | 诊断图输出子目录 |
| `diagnostics_sample_scale0` | True | 是否额外只采样 scale[0] 并 decode |

---

## 2. SRtrain.sh 套壳变量

入口：`SRtrain.sh`。脚本读取环境变量、给出默认值、展开成 `torchrun SRtrain.py ...`。常用变量：

| 类别 | 变量 | 默认 |
|---|---|---|
| 实验 | `EXP_NAME` / `EXP_NOTE` / `BED` / `PORT` | 必填 / `""` / `local_output` / `13333` |
| GPU | `CUDA_VISIBLE_DEVICES` | `0` |
| 数据 | `DATA_PATH` / `LR_FOLDER` / `HR_FOLDER` | 必填 / `LR_64x64` / `HR` |
| 多尺度 | `PATCH_NUMS_STR` | `"1 2 3 4 5 6 8 10 13 16"`（**必须**与 ckpt 一致） |
| VAE | `VAE_CKPT` / `CVAE` / `VAE_CH` / `QUANT_RESI` / `SHARE_QUANT_RESI` | 必填 / 32 / 128 / 0.5 / 4 |
| stage1 | `STAGE1_CKPT` / `LR_COND_SOURCE` / `SKIP_SCALE0_LOSS` | `""` / `srvar_encoder` / `False` |
| 训练 | `EP` / `BS` / `AC` / `LR` / `WD` / `WP` / `GRAD_CLIP` | 50 / 4 / 1 / 3e-4 / 0.05 / 0 / 2.0 |
| DiffLoss | `DIFFLOSS_W` / `DIFFLOSS_D` / `DIFF_STEPS` / `DIFFLOSS_BATCH_MUL` / `SCALE_LOSS_WEIGHTING` / `SCALE0_QUERY_SOURCE` | 1024 / 3 / `"100"` / 4 / `token` / `sos` |
| CFG | `CFG` / `CFG_INFER` | 0.1 / 1.0 |
| 验证/重建 | `VAL_AND_SAVING_PER_EP` / `RECON_SAVE_INTERVAL` / `RECON_MAX_SAMPLES` / `RECON_DIR_NAME` / `EVAL_AR_MAX_BATCHES` | 2 / 0 / 4 / `reconstruction_samples` / 4 |
| 日志 | `TRAIN_LOG_POINTS_PER_EPOCH` | 32 |
| 诊断 | `DIAGNOSTICS_ENABLED` / `DIAGNOSTICS_INTERVAL` / `DIAGNOSTICS_DIR_NAME` / `DIAGNOSTICS_SAMPLE_SCALE0` | True / 0 / `diagnostics` / True |

---

## 3. 训练循环

文件：`SRtrainer.py`

`train_step` 主体（每次都做）：

```python
ms_h_target, ms_x_input, _ = vae_local.img_to_ms_continuous_input(inp_HR)   # no_grad
loss = srvar(inp_LR, ms_h_target, ms_x_input, scale_schedule, ref_B3HW)
```

log iter 时（每 epoch 大约 `TRAIN_LOG_POINTS_PER_EPOCH` 次，默认 32）：

- `latent_mse_last`：用 backbone 当前 `z` + `diffloss.sample(z, cfg=1.0)` 跑一次 last scale，跟 `ms_h_target[-1]` 做 MSE，作为收敛代理。
- `train_psnr / train_ssim`（开关 `log_train_psnr`）：跑一次 `_quick_reconstruction`，与 GT HR 算 PSNR/SSIM。

`eval_ep`（每 `val_and_saving_per_ep` 个 epoch）：

- 计算 DiffLoss val loss。
- 对前 `eval_ar_max_batches` 个 batch 调 `srvar_wo_ddp.autoregressive_infer_cfg`，得到 `hr_pred`，与 GT HR 算 PSNR/SSIM，**allreduce 后**取平均。

---

## 4. 重建图与 run_metadata.json

文件：`utils/image_saver.py`（直接搬 myvaex 同名工具，**改 1 处**：`nrow=3` 三列 `LR_upsampled | HR_pred | HR_gt`）。

- 保存路径：`bed/{exp_name}/{reconstruction_dir_name}/epXXXX_itYYYYYY_comparison.png`
- 触发条件：`it == 0 or it in metric_lg.log_iters or (reconstruction_save_interval>0 and it % reconstruction_save_interval == 0)`
- `run_metadata.json` 字段：`stage_name="SRVAR continuous AR"`、`save_dir`、`filename_pattern`、`frequency_description`、`comparison_layout="3 columns per row: LR_upsampled | HR_pred | HR_gt"`、`max_samples_per_image`、`postprocess`、`args`（整个 `Args.state_dict()`）。

诊断图：

- 保存路径：`bed/{diagnostics_dir_name}/epXXXX_itYYYYYY_diagnostic.png`
- 触发条件：`diagnostics_enabled=True` 且 `it == 0 or it in metric_lg.log_iters or (diagnostics_interval>0 and it % diagnostics_interval == 0)`。
- 布局：`LR_upsampled | AR_full | AR_scale0_only | VAE_oracle | HR_gt`。
- `VAE_oracle`：HR 经冻结 VAE 得到 target latent 后直接 decode，是 VAE ckpt 自身重建上限。
- `AR_scale0_only`：只跑第 0 个 AR scale 的 `diffloss.sample`，把该 coarse latent 累积/上采样后 decode，用来定位 scale[0] 起步是否已经崩掉。
- stdout 中 `[diagnostics ...]` 打印 AR / scale0 / oracle PSNR-SSIM；`[diagnostics latent]` 打印 LR、target_s0、target_last、sample_s0、sample_last 的 shape/mean/std/min/max。

---

## 5. PSNR / SSIM

调用 `skimage.metrics`：

```python
inp_norm = (inp + 1.0) / 2.0; inp_norm = inp_norm.clamp(0, 1)
rec_norm = (rec + 1.0) / 2.0; rec_norm = rec_norm.clamp(0, 1)
psnr = peak_signal_noise_ratio(inp_norm, rec_norm, data_range=1.0)
ssim = structural_similarity(inp_norm, rec_norm, data_range=1.0, channel_axis=2)
```

与 myvaex 完全一致（RGB 通道，`[0,1]`，`data_range=1.0`）。

---

## 6. Checkpoint

每隔 `val_and_saving_per_ep` epoch 保存：

```python
state = {
  'epoch': ep,
  'iter': g_it,
  'trainer': SRVARTrainer.state_dict(),     # 包含 srvar_wo_ddp, vae_local, var_opt
  'args': args.state_dict(),
}
```

- `ckpt-last.pth` 持续覆盖
- `ckpt-best.pth` 跟踪 val PSNR 最高
- 恢复训练用 `--resume=path/to/ckpt.pth`

---

## 7. 排查顺序

1. 启动期 assert：`patch_nums` 与 stage2 ckpt 一致、`low_len` 与 `tlen` 兼容、`lr_cond_source` 与 `lr_folder` / `stage1_ckpt` 兼容。
2. 看启动日志：`[diffloss init]` 的 final layer norm 应为 0 或接近 0；`scale0_query_source` / `scale_loss_weighting` 应符合本次实验预期。
3. 看 `[KL Debug]` 风格的日志（连续 VAE 内部）：mean/logvar 是否漂移。
4. 看 `loss` 是不是 NaN：DiffLoss 内 `learn_sigma` 在不稳定时可能出 inf；首先检查 `target` 是否冻结、形状是否一致。
5. 先看 `VAE_oracle`：如果 oracle 都很差，优先查 stage2 VAE ckpt / `patch_nums` / `vae_ch`。
6. 再看 `AR_scale0_only`：如果 scale0-only 已经是噪声，优先查 scale[0] 起点、SOS expansion、Plan A(stage1 LR_VAE)。
7. 最后看 `AR_full`：如果 scale0-only 还行但 full AR 崩，优先查后续 scale teacher forcing / `get_next_autoregressive_input` / DiffLoss sampling。
8. eval 期间 `autoregressive_infer_cfg` 比训练慢一个量级（每 token 100 步 DDPM），用 `eval_ar_max_batches` 控成本。
