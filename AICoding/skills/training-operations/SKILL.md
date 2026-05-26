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

### 1.6 Resume / BED / auto_resume

| 参数 | 默认 | 说明 |
|---|---|---|
| `local_out_dir_path` | `local_output` | 实验输出根目录；`SRtrain.sh` 环境变量为 **`BED`** |
| `resume` | `""` | 显式指定要加载的 checkpoint 路径（`SRtrain.sh` 环境变量为 **`RESUME`**） |
| `auto_resume` | True | 仅当 `resume` 为空时生效：在 `BED` 下按修改时间取最新的 `ar-ckpt*.pth` |

**恢复优先级**（实现：`utils/misc.py::auto_resume`，`SRtrain.py` 固定 pattern=`ar-ckpt*.pth`）：

| 优先级 | 条件 | 行为 |
|---|---|---|
| 1 | `--resume` 非空 | 从该路径加载；**不**在 `BED` 里搜索 |
| 2 | `--resume` 为空 且 `auto_resume=True` | 在 `{BED}/ar-ckpt*.pth` 中取 mtime 最新者 |
| 3 | 其余 | 不恢复，`start_ep=0, start_it=0` |

**`BED` 与 resume 解耦**：`--resume` 只决定**从哪里读权重**；新 ckpt / 日志 / 重建图仍写到当前 `BED`。可从 run A 的 ckpt 恢复，同时把输出写到 run B 的目录。

**自动恢复的范围**：glob 只匹配 `ar-ckpt*.pth`（如 `ar-ckpt-last.pth`、`ar-ckpt-best.pth`），**不会**自动选中 `ckpt-{ep}.pth`；若要 resume 后者，必须显式 `--resume=.../ckpt-8.pth`（注意文件名是 `ckpt-2.pth` 而非 `ar-ckpt-2.pth`）。

**显式 `--resume` 失败即退出**：路径不存在或 `torch.load` 失败时会 `raise`，避免误从头训练。自动 glob 找不到 ckpt 时仍静默从 ep0 开始。

**`BED` 还负责**（与 resume 无关）：`log.txt`、`stdout.txt`/`stderr.txt`、TensorBoard 子目录、`{reconstruction_dir_name}/`、`{diagnostics_dir_name}/`、周期性保存的 `ar-ckpt-last.pth` / `ar-ckpt-best.pth` / `ckpt-{ep}.pth`。

示例：

```bash
# 从指定 ckpt 恢复，输出仍写到当前 BED
RESUME=/path/to/old_run/ar-ckpt-last.pth \
BED=local_output/my_new_run \
bash SRtrain.sh

# 在 BED 内自动找最新 ar-ckpt*.pth（默认行为）
BED=local_output/my_run AUTO_RESUME=True bash SRtrain.sh

# 完全从头训练
AUTO_RESUME=False RESUME= bash SRtrain.sh
```

**易错**：多行 `VAR=val \` 命令若某行末尾缺 `\`，后续变量不会传给 `bash SRtrain.sh`；`BED=...` 行后建议也加 `\`。启动后核对日志里 `data_path`、`lr_folder`、`resume`、`local_out_dir_path` 是否与预期一致。

---

## 2. SRtrain.sh 套壳变量

入口：`SRtrain.sh`。脚本读取环境变量、给出默认值、展开成 `torchrun SRtrain.py ...`。常用变量：

| 类别 | 变量 | 默认 |
|---|---|---|
| 实验 | `EXP_NAME` / `EXP_NOTE` / `BED` / `PORT` | 必填 / `""` / `local_output` / `13333` |
| GPU | `CUDA_VISIBLE_DEVICES` | `0` |
| 数据 | `DATA_PATH` / `LR_FOLDER` / `HR_FOLDER` / `IMG_CHANNELS` | 必填 / `LR_64x64` / `HR` / `3` |
| 多尺度 | `PATCH_NUMS_STR` | `"1 2 3 4 5 6 8 10 13 16"`（**必须**与 ckpt 一致） |
| VAE | `VAE_CKPT` / `CVAE` / `VAE_CH` / `QUANT_RESI` / `SHARE_QUANT_RESI` | 必填 / 32 / 128 / 0.5 / 4 |
| stage1 | `STAGE1_CKPT` / `LR_COND_SOURCE` / `SKIP_SCALE0_LOSS` | `""` / `srvar_encoder` / `False` |
| 训练 | `EP` / `BS` / `AC` / `LR` / `WD` / `WP` / `GRAD_CLIP` | 50 / 4 / 1 / 3e-4 / 0.05 / 0 / 2.0 |
| DiffLoss | `DIFFLOSS_W` / `DIFFLOSS_D` / `DIFF_STEPS` / `DIFFLOSS_BATCH_MUL` / `SCALE_LOSS_WEIGHTING` / `SCALE0_QUERY_SOURCE` | 1024 / 3 / `"100"` / 4 / `token` / `sos` |
| CFG | `CFG` / `CFG_INFER` | 0.1 / 1.0 |
| 验证/重建 | `VAL_AND_SAVING_PER_EP` / `RECON_SAVE_INTERVAL` / `RECON_MAX_SAMPLES` / `RECON_DIR_NAME` / `EVAL_AR_MAX_BATCHES` | 2 / 0 / 4 / `reconstruction_samples` / 4 |
| 日志 | `TRAIN_LOG_POINTS_PER_EPOCH` | 32 |
| 诊断 | `DIAGNOSTICS_ENABLED` / `DIAGNOSTICS_INTERVAL` / `DIAGNOSTICS_DIR_NAME` / `DIAGNOSTICS_SAMPLE_SCALE0` | True / 0 / `diagnostics` / True |
| 恢复 | `RESUME` / `AUTO_RESUME` | `""` / `True` |

`IMG_CHANNELS=1` 用于单通道医学灰度图；`SRtrain.py` 会优先读取 `VAE_CKPT` 里的 `args.img_channels`，不一致时自动覆盖命令行值，避免上游 VAE 权重通道形状不匹配。

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

- 保存路径：`{BED}/{reconstruction_dir_name}/epXXXX_itYYYYYY_comparison.png`（`EXP_NAME` 不参与路径）
- 触发条件：`it == 0 or it in metric_lg.log_iters or (reconstruction_save_interval>0 and it % reconstruction_save_interval == 0)`
- `run_metadata.json` 字段：`stage_name="SRVAR continuous AR"`、`save_dir`、`filename_pattern`、`frequency_description`、`comparison_layout="3 columns per row: LR_upsampled | HR_pred | HR_gt"`、`max_samples_per_image`、`postprocess`、`args`（整个 `Args.state_dict()`）。

诊断图：

- 保存路径：`{BED}/{diagnostics_dir_name}/epXXXX_itYYYYYY_diagnostic.png`
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
ssim = structural_similarity(inp_norm, rec_norm, data_range=1.0, channel_axis=2)  # RGB
# grayscale C=1 uses HW arrays and omits channel_axis
```

与 myvaex 完全一致：`C=3` 时使用 RGB/HWC + `channel_axis=2`；`IMG_CHANNELS=1` 时使用灰度 `H x W`，仍然是 `[0,1]`、`data_range=1.0`。

---

## 6. Checkpoint 与恢复训练

每隔 `val_and_saving_per_ep` epoch 保存（`SRtrain.py`，仅 local master）：

```python
state = {
  'epoch': ep + 1,
  'iter': 0,
  'trainer': SRVARTrainer.state_dict(),   # srvar_wo_ddp, vae_local, var_opt；加载时 skip_vae=True
  'args': args.state_dict(),
}
```

| 文件名 | 说明 |
|---|---|
| `ar-ckpt-last.pth` | 每次验证后覆盖，**自动 resume 默认匹配此模式** |
| `ar-ckpt-best.pth` | val diff loss 最优时从 last 复制 |
| `ckpt-{ep}.pth` | 每 epoch 额外留档；**不会**被 `auto_resume` 自动选中 |

恢复逻辑见 **§1.6**。`trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)`——冻结 VAE 权重以当前 `--vae_ckpt` 为准，不沿用 ckpt 内 VAE。

---


## 7. 排查顺序

1. 启动期 assert：`patch_nums` 与 stage2 ckpt 一致、`low_len` 与 `tlen` 兼容、`lr_cond_source` 与 `lr_folder` / `stage1_ckpt` 兼容；核对 `[auto_resume]` 日志是否从预期的 `--resume` 或 `BED` 路径加载。
2. 看启动日志：`[diffloss init]` 的 final layer norm 应为 0 或接近 0；`scale0_query_source` / `scale_loss_weighting` 应符合本次实验预期。
3. 看 `[KL Debug]` 风格的日志（连续 VAE 内部）：mean/logvar 是否漂移。
4. 看 `loss` 是不是 NaN：DiffLoss 内 `learn_sigma` 在不稳定时可能出 inf；首先检查 `target` 是否冻结、形状是否一致。
5. 先看 `VAE_oracle`：如果 oracle 都很差，优先查 stage2 VAE ckpt / `patch_nums` / `vae_ch`。
6. 再看 `AR_scale0_only`：如果 scale0-only 已经是噪声，优先查 scale[0] 起点、SOS expansion、Plan A(stage1 LR_VAE)。
7. 最后看 `AR_full`：如果 scale0-only 还行但 full AR 崩，优先查后续 scale teacher forcing / `get_next_autoregressive_input` / DiffLoss sampling。
8. eval 期间 `autoregressive_infer_cfg` 比训练慢一个量级（每 token 100 步 DDPM），用 `eval_ar_max_batches` 控成本。
