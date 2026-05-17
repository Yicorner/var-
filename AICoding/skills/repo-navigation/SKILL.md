---
name: repo-navigation
description: Route readers to the correct var/ skill before they dive into detailed project, architecture, training or LR-data documentation.
---

# Repo Navigation (var)

> 面向 AI 助手和维护者的仓库导航页。这里不负责讲项目细节，只负责把你带到正确的 skill。var 是 myvaex 项目的下游：在 myvaex 训练好的 **多尺度连续 VAE** 上做条件式自回归超分。

---

## 1. 使用方式

- 先读本文件，判断当前问题属于哪一类。
- 再进入对应专题 skill，避免在一个超长文档里来回搜索。
- 与上游 myvaex 共用的工程约定（数据子目录、PSNR/SSIM 算法、`save_reconstruction_comparison` 等），优先参考 `myvaex/AICoding/skills/`。

---

## 2. Skill 导航

### 2.1 项目整体结构与目标

文件夹：`AICoding/skills/project-overview/`

适合回答：

- 这个项目整体在做什么、与 myvaex 是什么关系
- `models/`、`utils/`、`SRtrainer.py`、`SRtrain.py` 的职责边界
- 与 myvaex stage2 ckpt 的接口契约（`patch_nums`、`Cvae`、`ch`、`quant_resi`、`share_quant_resi` 必须严格一致）

---

### 2.2 SRVAR 架构与张量流

文件夹：`AICoding/skills/architecture/`

适合回答：

- SRVAR 的 forward 与 `autoregressive_infer_cfg` 张量形状
- SOS / Block-causal Self-Attention / Cross-Attention 与 LR condition 的关系
- 训练时 teacher forcing 用的多尺度连续 latent 是怎么构造的

---

### 2.3 连续 AR Head（MAR 范式 DiffLoss）

文件夹：`AICoding/skills/continuous-ar-head/`

适合回答：

- 为什么 var 不再用 CrossEntropy 头
- DiffLoss 怎么接到 SRVAR：`target_channels=Cvae`、`z_channels=embed_dim`
- 训练用 1000 步、推理用 spaced (e.g. 100)；CFG 的 `forward_with_cfg`
- `diffloss_batch_mul` 的作用

---

### 2.4 LR 数据约定与开关耦合

文件夹：`AICoding/skills/lr-data-conventions/`

适合回答：

- `LR_64x64` vs `LR_256` 怎么选
- `--lr_folder` / `--lr_cond_source` / `--stage1_ckpt` 三个开关的语义矩阵
- 与 `patch_nums[0]` 的耦合关系（启用方案 A 时 `patch_nums[0]` 必须是 LR_VAE latent 边长）

---

### 2.5 训练运维与产物

文件夹：`AICoding/skills/training-operations/`

适合回答：

- `SRtrain.sh` 套壳变量都有哪些
- 训练循环、log iter、`save_reconstruction_comparison` + `run_metadata.json` 怎么生成
- PSNR / SSIM 在哪算（与 myvaex 一致，RGB 0-1，skimage）

---

## 3. 维护规则

- 这里保留索引、入口和最小摘要，不再堆叠实现细节。
- 与 myvaex 共享的约定（PairedImageDataset 命名约定、PSNR/SSIM 度量管线）写在 myvaex 那边，本仓只放 var 特有差异。
- 新增专题时，优先保持「一个 skill 解决一类问题」。
