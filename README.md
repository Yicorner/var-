# var (SRVAR continuous AR)

> 训前步骤
>
> 1. `conda activate /home/featurize/work/myenv` （与 myvaex 共用环境，依赖一致）
> 2. 准备 myvaex stage2 的连续多尺度 VAE checkpoint（必填，作为 `--vae_ckpt`）
> 3. 跑 `SRtrain.sh` 脚本

`var/` 是 `myvaex` 的下游：在 myvaex 训好的 **多尺度连续 VAE 的 latent 空间** 上做 **条件式自回归超分**，token 头采用 MAR 范式的 `DiffLoss`（每个位置一次小扩散）。

详细文档请优先看仓内 SKILL：

- 总览：`AICoding/skills/project-overview/`
- 架构与张量流：`AICoding/skills/architecture/`
- 连续 AR 头：`AICoding/skills/continuous-ar-head/`
- LR 数据约定：`AICoding/skills/lr-data-conventions/`
- 训练运维：`AICoding/skills/training-operations/`

---

## 1. 任务定义

- 输入：LR（默认 `LR_64x64`；也支持上游用 trilinear 上采样到 256×256 的 `LR`）
- 输出：HR（256×256）
- 训练目标：在 myvaex stage2 VAE 的多尺度连续 latent 上做自回归预测，每个空间位置上由 `DiffLoss(SimpleMLPAdaLN)` 跑一次 IDDPM；不再有离散 vocabulary 或 CrossEntropy。

---

## 2. 与 myvaex 的依赖契约

`--vae_ckpt` 必须由 myvaex stage2 训出。**以下五个参数必须与 stage2 ckpt 严格一致**（不一致时 `load_state_dict(..., strict=True)` 会直接报错，或运行时 `quant_resi` ticks 偏移导致重建乱码）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `patch_nums` | `(1,2,3,4,5,6,8,10,13,16)` | 多尺度配置 |
| `Cvae` (`--Ct5`) | `32` | latent 通道数 |
| `vae_ch` | `128` | encoder/decoder 基础通道数 |
| `quant_resi` | `0.5` | residual conv 比例 |
| `share_quant_resi` | `4` | quant_resi 共享方式 |

启动时 `SRtrain.py` 会做 assert：

```python
assert tuple(vae_local.quantize.v_patch_nums) == tuple(args.patch_nums)
```

---

## 3. 三种典型实验脚本

### 3.1 默认（**方案 B + LR_64x64 + srvar_encoder**，推荐起点）

```bash
EXP_NAME=srvar_default \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
VAE_CKPT=local_output/stage2/ckpt-best.pth \
PATCH_NUMS_STR="1 2 3 4 5 6 8 10 13 16" \
bash SRtrain.sh
```

- `scale[0]` 仍由 SOS 预测；不依赖 stage1 LR_VAE。
- `low_f` 来自 SRVAR 内部 encoder（权重从冻结 VAE 拷过来）。
- KV 长度 = 16（`LR_64x64 / 16 = 4×4`）。

### 3.2 沿用旧 var 行为（**LR_256 + srvar_encoder**，对照实验）

```bash
EXP_NAME=srvar_lr256_baseline \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=local_output/stage2/ckpt-best.pth \
PATCH_NUMS_STR="1 2 3 4 5 6 8 10 13 16" \
bash SRtrain.sh
```

- KV 长度 = 256（`LR_256 / 16 = 16×16`）；信息量与 LR_64x64 相同但 KV 更长，成本更高。

### 3.3 方案 A（**Stage1 LR_VAE → 4×4 prior + 跳过 scale[0] loss**）

```bash
EXP_NAME=srvar_stage1_planA \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
VAE_CKPT=local_output/stage2/ckpt-best.pth \
STAGE1_CKPT=local_output/stage1/ckpt-best.pth \
LR_COND_SOURCE=lr_vae \
SKIP_SCALE0_LOSS=True \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
LR_FOLDER=LR_64x64 \
bash SRtrain.sh
```

- 启动时强 assert：`patch_nums[0] == lr_vae.encode(LR_64x64).shape[-1]`（默认 4）。
- `ms_h_target[0]` 用 LR_VAE 的 `posterior.mode()` 覆盖；`--skip_scale0_loss=True` 会让 DiffLoss 不学最小尺度。
- KV 由 LR_VAE 提供（4×4 = 16 个 token）。

---

## 4. 推理 / 指标

```bash
python metric.py \
  --ckpt local_output/srvar_default/ar-ckpt-best.pth \
  --data_path /path/to/dataset/val \
  --cfg 1.0 \
  --diff_steps 100
```

输出：

- 每张预测图 + `LR_upsampled | HR_pred | HR_gt` 三联对比图（`reconstruction_samples/`）
- `metrics.csv`（PSNR / SSIM 与 pyiqa）
- `run_metadata.json`（实验参数、保存频率、后处理流程）

PSNR / SSIM 与 myvaex 完全一致：RGB 通道、[0,1]、`skimage.metrics`，`data_range=1.0`。

---

## 5. 已知约束

| 项目 | 约束 |
|---|---|
| `patch_nums` | 必须与 stage2 ckpt 一致 |
| `lr_folder` × `lr_cond_source` | `LR_COND_SOURCE=lr_vae` 强制 `LR_FOLDER=LR_64x64` 且 `STAGE1_CKPT` 非空 |
| `tlen` | `cfg_uncond` buffer 长度，需 `>= low_len`（`LR_64`→16，`LR_256`→256） |
| `patch_nums[0]` | 非 stage1 时可以是 `1` 或 `4`；SRVAR 会展开 `patch_nums[0]^2` 个 SOS 起始 token。启用 stage1 时必须等于 `lr_img_size/16`（默认 4） |
| DiffLoss train / gen | 训练 1000 步 cosine IDDPM；推理 spaced 步数 (默认 100) |
| `same_shape` | 默认 `False`，LR 原样喂入；`True` 会把 LR bicubic 上采样到 HR，仅 legacy 兼容 |

---

## 6. 重建可视化与 run_metadata.json

每个 log iter 写一张 `epXXXX_itYYYYYY_comparison.png`：**3 列** `LR_upsampled | HR_pred | HR_gt`，每行一个样本。
`run_metadata.json` 在第一次落盘时写入，包括：

- `stage_name="SRVAR continuous AR (DiffLoss head)"`
- `comparison_layout`：解释 3 列的含义
- `args`：本次训练参数的完整快照
- `postprocess`：denormalize / clamp / upsample / grid 的完整后处理链

### 6.1 诊断图与更高频日志

默认 `TRAIN_LOG_POINTS_PER_EPOCH=32`，会比之前更频繁地打印训练日志。诊断输出默认开启：

```bash
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=0 \
DIAGNOSTICS_DIR_NAME=diagnostics \
bash SRtrain.sh
```

- `DIAGNOSTICS_INTERVAL=0`：跟随 train log iter；设成 `500` 表示每 500 iter 额外诊断一次。
- 诊断图是 **5 列**：`LR_upsampled | AR_full | AR_scale0_only | VAE_oracle | HR_gt`。
- `VAE_oracle` 是用 HR 经冻结 VAE 得到的 target latent 直接 decode，代表这份 VAE ckpt 的重建上限。
- `AR_scale0_only` 是只跑第 0 个 AR scale 的 diffusion sampling，把这个 coarse latent 上采样/累积后直接 decode，用来判断 scale[0] 起步是否已经坏掉。
- stdout 会打印 `[diagnostics ...]` 和 `[diagnostics latent]`，包含 AR / scale0 / oracle 的 PSNR、SSIM 和 target/sample latent 的 mean/std/min/max。
- 启动日志会打印 `scale0_query_source`、`scale_loss_weighting` 和 DiffLoss final layer norm；新实验建议检查 final layer norm 为 0 或接近 0。

---

## 7. 与 myvaex stage1/stage2 ckpt 的兼容性

- `vae_ckpt` 优先从 `ckpt['trainer']['vae_ema']` 读取，fallback 到 `vae_wo_ddp` / `vae`。
- `stage1_ckpt` 优先从 `ckpt['trainer']['lr_vae_ema']` 读取，fallback 到 `lr_vae_wo_ddp` / `lr_vae`。
- 无 `trainer` 顶层 key 时，按裸 `state_dict` 处理（兼容 `eval_stage1_ckpt.py` 的导出格式）。



## 8. LR_256 baseline quick command

```bash
EXP_NAME=srvar_lr256_baseline \
EXP_NOTE="cond and scale[0] don't depend on LR_VAE" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=cond_and_scale[0]_dont_depend_on_LR_VAE \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
bash SRtrain.sh
```

### 8.1 Round3 推荐短跑

这条命令启用 LR-conditioned scale0 query、equal-scale DiffLoss weighting，并用 gradient accumulation 保持 microbatch 约 4、实际 peak `tlr≈1e-4`。不要从旧噪声 run resume。

```bash
EXP_NAME=srvar_scale0_lrq_equal_loss \
EXP_NOTE="LR-conditioned scale0 query + equal-scale DiffLoss" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=scale0_lrq_equal_loss \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
BS=256 \
AC=64 \
LR=1e-4 \
WP=0.05 \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=equal_scale \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_DIR_NAME=scale0_lrq_equal_loss/diagnostics \
bash SRtrain.sh
```

```bash
EXP_NAME=srvar_lr256_baseline_Diagnostics \
EXP_NOTE="cond and scale[0] don't depend on LR_VAE with diagnostics" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=cond_and_scale[0]_dont_depend_on_LR_VAE_with_diagnostics \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_DIR_NAME=cond_and_scale[0]_dont_depend_on_LR_VAE_with_diagnostics/diagnostics \
bash SRtrain.sh
```

```bash
EXP_NAME=srvar_scale0_lrq_equal_loss_fix_grad_bug \
EXP_NOTE="LR-conditioned scale0 query + equal-scale DiffLoss + fix zero grad bug + clipped DiffLoss sampling diagnostic" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=srvar_scale0_lrq_equal_loss_fix_grad_bug \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
BS=64 \
AC=16 \
LR=4e-4 \
WP=0.05 \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=equal_scale \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_DIR_NAME=srvar_scale0_lrq_equal_loss_fix_grad_bug/diagnostics \
bash SRtrain.sh
```

```bash
EXP_NAME=srvar_mse_head_scale0_sanity \
EXP_NOTE="direct MSE latent head sanity check for LR-conditioned scale0" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=mse_head_scale0_sanity \
DIAGNOSTICS_DIR_NAME=mse_head_scale0_sanity/diagnostics \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
BS=64 \
AC=16 \
LR=4e-4 \
WP=0.05 \
CONTINUOUS_HEAD_TYPE=mse \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=equal_scale \
DIFFLOSS_SAMPLE_CLIP_DENOISED=True \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_SAMPLE_SCALE0=True \
DIAGNOSTICS_DIR_NAME=mse_head_scale0_sanity/diagnostics \
bash SRtrain.sh
```

```bash
EXP_NAME=srvar_mse_head_scale0_sanity_per-token_loss \
EXP_NOTE="direct MSE latent head sanity check for LR-conditioned scale0 + per-token_loss" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=srvar_mse_head_scale0_sanity_per-token_loss \
DIAGNOSTICS_DIR_NAME=srvar_mse_head_scale0_sanity_per-token_loss/diagnostics \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
BS=64 \
AC=16 \
LR=4e-4 \
WP=0.05 \
CONTINUOUS_HEAD_TYPE=mse \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=token \
DIFFLOSS_SAMPLE_CLIP_DENOISED=True \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_SAMPLE_SCALE0=True \
bash SRtrain.sh
```

```bash
EXP_NAME=srvar_mse_token_cfg0_long \
EXP_NOTE="MSE token loss, no CFG dropout, longer run for late scales, resume from CFG=0.1" \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=mse_token_cfg0_long \
VAL_AND_SAVING_PER_EP=2 \
VAE_CH=160 \
BS=64 \
AC=16 \
LR=4e-4 \
WP=0.05 \
CFG=0.0 \
CONTINUOUS_HEAD_TYPE=mse \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=token \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_DIR_NAME=mse_token_cfg0_long/diagnostics \
EP=8 \
bash SRtrain.sh
```

## 9. 单通道医学影像模式

`var/` 的 `IMG_CHANNELS` 必须和上游 myvaex stage2 VAE checkpoint 一致。默认是 `IMG_CHANNELS=3` 以兼容旧 RGB checkpoint；灰度医学影像实验请使用 `IMG_CHANNELS=1`，并传入同样用 `IMG_CHANNELS=1` 训练出来的 myvaex stage2 checkpoint。

`SRtrain.py` 会优先读取 `VAE_CKPT` 中保存的 `args.img_channels`，如果和命令行不同，会自动覆盖成 checkpoint 的通道数，避免 VAE 权重形状对不上。数据加载、SRVAR 内部 encoder、SRVAR 推理解码都会跟随该通道数；PSNR / SSIM 在 `C=1` 时按真正灰度图计算。

推荐灰度 SRVAR 命令：

```bash
EXP_NAME=srvar_gray_scale0_lrq_equal_loss \
EXP_NOTE="single-channel SRVAR, LR-conditioned scale0 query + equal-scale DiffLoss" \
IMG_CHANNELS=1 \
LR_FOLDER=LR \
SAME_SHAPE=False \
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
VAE_CKPT=/home/featurize/work/myvaex/local_output/test/your_gray_stage2/ckpt-best.pth \
PATCH_NUMS_STR="4 5 6 8 10 13 16" \
RECON_DIR_NAME=srvar_gray_scale0_lrq_equal_loss \
VAL_AND_SAVING_PER_EP=1 \
VAE_CH=160 \
BS=64 \
AC=16 \
LR=4e-4 \
WP=0.05 \
SCALE0_QUERY_SOURCE=low_f_pool \
SCALE_LOSS_WEIGHTING=equal_scale \
TRAIN_LOG_POINTS_PER_EPOCH=64 \
DIAGNOSTICS_ENABLED=True \
DIAGNOSTICS_INTERVAL=500 \
DIAGNOSTICS_SAMPLE_SCALE0=True \
bash SRtrain.sh
```
