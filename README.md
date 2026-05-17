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
| `patch_nums[0]` | 启用 stage1 时必须等于 `lr_img_size/16`（默认 4） |
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

---

## 7. 与 myvaex stage1/stage2 ckpt 的兼容性

- `vae_ckpt` 优先从 `ckpt['trainer']['vae_ema']` 读取，fallback 到 `vae_wo_ddp` / `vae`。
- `stage1_ckpt` 优先从 `ckpt['trainer']['lr_vae_ema']` 读取，fallback 到 `lr_vae_wo_ddp` / `lr_vae`。
- 无 `trainer` 顶层 key 时，按裸 `state_dict` 处理（兼容 `eval_stage1_ckpt.py` 的导出格式）。
