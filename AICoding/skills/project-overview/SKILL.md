---
name: project-overview
description: Explain what the var/ repository builds, where the core modules live, and how it depends on the upstream myvaex multi-scale continuous VAE.
---

# Project Overview (var)

> 面向第一次进入仓库的人。目标是快速建立对「项目在做什么、关键代码在哪里、模块如何分工」的整体认识。

---

## 1. 项目在做什么

`var/` 是一个 **多尺度自回归医学影像超分模型**：

- 输入：LR（低分辨率，例如 64×64 或 256×256 经 trilinear 上采样而来的低质图）。
- 输出：HR（256×256）。
- 思路：在 **myvaex 训练好的多尺度连续 VAE 的 latent 空间** 上做自回归。

历史演化：

- 早期：HR 用离散 VQ-VAE，VAR 在 codebook index 上做 CrossEntropy 自回归。
- 当前：HR 用 myvaex 的 **连续多尺度 VAE**（无 codebook，Gaussian posterior），var 在连续 latent 上做 AR；token 头改为 **MAR 范式的 DiffLoss**（每个位置一次小扩散）。

---

## 2. 与 myvaex 的依赖

var **完全依赖 myvaex stage2 训练出的多尺度连续 VAE checkpoint**：

- 训练 var 时 VAE 全程冻结。
- VAE 的 `patch_nums`、`Cvae`、`ch`、`quant_resi`、`share_quant_resi` 必须与 var 启动参数严格一致；不一致会导致 `quant_resi[si/(SN-1)]` 的 `PhiPartiallyShared` ticks 索引偏移，重建结果完全错乱。
- `--stage1_ckpt` 非空时，**额外**加载 myvaex stage1 LR_VAE 用作 `scale[0]` 的 prior。

---

## 3. 关键目录

```text
var/
├── AICoding/skills/         # 本 SKILL 文档体系
├── models/
│   ├── __init__.py          # build_vae_srvar(), 可选 build_lr_vae()
│   ├── SRVAR.py             # 主模型：encoder(LR) + SOS + SA/CrossAttn + DiffLoss head
│   ├── var.py               # 原始 ImageNet 类条件 VAR（保留，已不再训练）
│   ├── vqvae.py             # 连续多尺度 VAE（沿用 myvaex 的实现）
│   ├── quant.py             # ContinuousMultiScaleQuantizer + DiagonalGaussianDistribution
│   ├── lr_vae.py            # LR 单尺度 VAE（test_mode=True，仅推断用）
│   ├── basic.py             # CrossAttn / SelfAttn / RMSNorm / RoPE2D
│   ├── basic_var.py         # AdaLNSelfAttn / AdaLNBeforeHead（部分保留）
│   ├── basic_vae.py         # Encoder/Decoder 基础 CNN（与 myvaex 同源）
│   ├── helpers.py           # gumbel/top-k 等采样工具（部分已退役）
│   ├── flex_attn.py         # 可选 PyTorch 2.5 flex_attention 封装
│   ├── fused_op.py          # @torch.compile 的 fused RMSNorm / AdaLN
│   ├── Unet/                # 仅 DiffLoss 内部使用的 UNet（旧路径，可移除）
│   └── diffusion/
│       ├── diffloss.py      # MAR 风格 DiffLoss(SimpleMLPAdaLN)
│       ├── gaussian_diffusion.py / respace.py / diffusion_utils.py
├── utils/
│   ├── arg_util.py          # var 的训练参数
│   ├── data.py              # PairedImageDataset (lr_folder / hr_folder 参数化)
│   ├── data_sampler.py
│   ├── dynamic_resolution.py
│   ├── amp_sc.py            # AmpOptimizer 封装
│   ├── image_saver.py       # save_reconstruction_comparison + run_metadata
│   ├── lr_control.py
│   └── misc.py
├── SRtrain.py               # 训练入口（torchrun）
├── SRtrainer.py             # SRVARTrainer：DiffLoss 主 loss、PSNR/SSIM 评估
├── SRtrain.sh               # 套壳脚本（环境变量优先）
├── metric.py                # 推理 + pyiqa 指标
├── README.md
└── test.py                  # tensorboard 调试片段
```

---

## 4. 核心模型

### 4.1 多尺度连续 VAE（冻结）

文件：`models/vqvae.py` + `models/quant.py`

职责：把 HR 图像分解为多个尺度的连续 latent。

关键接口：

- `vae.img_to_ms_continuous_input(img_HR) -> (ms_h_target, ms_x_input, f_hat_full)`
  - `ms_h_target`: list of `[B, C, pn, pn]`，每个 scale 的 `posterior.mode()`，作为 DiffLoss 目标。
  - `ms_x_input`: `[B, L-1, C]`，teacher forcing 输入（与离散版 `idxBl_to_var_input` 同形）。
- `vae.fhat_to_img(f_hat) -> [-1,1]` 图像。
- `vae.quantize.get_next_autoregressive_input(si, SN, accu_BChw, h_BChw)`：推理用，累积 + 下采样到下一尺度。

### 4.2 SRVAR

文件：`models/SRVAR.py`

职责：在连续 latent 上做条件多尺度 AR。结构：

```
LR --> encoder/quant_conv  (或 LR_VAE) --> low_f [B, low_len, C]
                                            |
                                            +--> TextAttentivePool -> SOS / cond_BD
                                            +--> low_proj_for_ca -> CrossAttn KV

ms_x_input [B, L-1, C] -> word_embed -> [B, L-1, D]
SOS [B,1,D] cat embed -> [B, L, D] -> Self-Attn (block-causal) + CrossAttn -> z [B, L, D]

z -> reshape(B*L, D) ----+
ms_h_target -> [B*L, C] -+--> DiffLoss(target, z) -> scalar loss
```

### 4.3 DiffLoss Head

文件：`models/diffusion/diffloss.py`

- `forward(target, z) -> scalar loss`：训练用 1000 步 IDDPM。
- `sample(z, temperature, cfg) -> [N, C]`：推理用 spaced 步数（默认 100），ancestral sampling。
- `target_channels = Cvae`，`z_channels = embed_dim`。

### 4.4 LR_VAE（可选）

文件：`models/lr_vae.py`

- 仅在 `--stage1_ckpt` 非空时构造，全程冻结。
- 输入 LR_64x64 → `[B, C, 4, 4]` posterior mean，可作为 `scale[0]` 的 prior 或 cross-attention 的 KV。

---

## 5. 进入代码时的优先顺序

1. `utils/arg_util.py`：确认入口参数与默认值（特别是 `patch_nums`、`vae_ckpt`、`stage1_ckpt`、`lr_folder`、`lr_cond_source`）。
2. `models/SRVAR.py` 的 `forward` 与 `autoregressive_infer_cfg`：理解张量流。
3. `SRtrainer.py` 的 `train_step` 与 `eval_ep`：理解主 loss、辅助指标与重建可视化触发点。
4. `SRtrain.sh`：理解外部如何套壳常用参数。

---

## 6. 相关 skill

- 架构与张量流：`AICoding/skills/architecture/`
- 连续 AR 头：`AICoding/skills/continuous-ar-head/`
- LR 数据约定：`AICoding/skills/lr-data-conventions/`
- 训练运维：`AICoding/skills/training-operations/`
- 上游 myvaex 的训练与可视化约定：`myvaex/AICoding/skills/`
