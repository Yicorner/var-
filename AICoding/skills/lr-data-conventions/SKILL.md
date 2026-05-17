---
name: lr-data-conventions
description: Document the LR data flow choices (LR_64x64 vs LR_256), the --lr_folder / --lr_cond_source / --stage1_ckpt switch matrix, and their coupling with patch_nums.
---

# LR Data Conventions

> 面向需要切换 LR 输入分辨率、cross-attention 条件来源、是否启用 stage1 LR_VAE 的人。把三个开关之间的相互约束写清楚。

---

## 1. 数据集布局

```text
DATA_PATH/
├── train/
│   ├── LR_64x64/    # 原始 64x64 低分辨率
│   ├── LR/          # LR_64x64 trilinear 上采样到 256x256（信息 ≤ LR_64x64）
│   └── HR/          # 256x256 高分辨率
└── val/
    ├── LR_64x64/
    ├── LR/
    └── HR/
```

**信息量关系**：`LR_64x64 ≥ LR_256`。`LR_256` 是 `LR_64x64` 经过插值生成的，没有新信息但损失高频边缘。
**默认推荐**：`--lr_folder=LR_64x64`，与 myvaex stage1 保持一致。

---

## 2. 三个开关

### 2.1 `--lr_folder` (字符串)

每个 split 下 LR 子目录名。默认 `LR_64x64`。`PairedImageDataset` 直接拼出 `DATA_PATH/<split>/<lr_folder>` 与 `DATA_PATH/<split>/<hr_folder>` 配对。

### 2.2 `--lr_cond_source` (枚举: `srvar_encoder` | `lr_vae`)

控制 `low_f` 来自哪个 encoder：

- `srvar_encoder`：SRVAR 自带 `encoder + quant_conv`（权重通过 `init_LREncoder` 从冻结 HR VAE 复制），跟 SRVAR 一起训练。
- `lr_vae`：直接用 stage1 LR_VAE 的 `encode_to_posterior_mean()`。SRVAR 不构造自己的 encoder。

### 2.3 `--stage1_ckpt` (路径，可空)

`""` 表示不使用 stage1（方案 B，默认）。
非空表示加载 myvaex stage1 LR_VAE：

- 同时 `--lr_cond_source=lr_vae` 时，LR_VAE 既给 cross-attn KV、也给 `scale[0]` 的 prior（覆盖 `ms_h_target[0]`）。
- 仅在 `--lr_cond_source=srvar_encoder` 时，LR_VAE 只参与 `scale[0]` prior 覆盖，不参与 cross-attn。

---

## 3. 语义矩阵（必须 assert 的所有合法组合）

| `lr_folder` | `lr_cond_source` | `stage1_ckpt` | low_len | scale[0] 起点 | 行为说明 |
|---|---|---|---|---|---|
| `LR_64x64` (默认) | `srvar_encoder` (默认) | 空 | 16 (4×4) | SOS 预测 | **方案 B 默认**：信息最全 + 改动最小 |
| `LR_64x64` | `lr_vae` | 必填 | 16 (4×4) | LR_VAE mean | **方案 A**：scale[0] 用 LR_VAE 的 4×4 latent 直接作为 prior |
| `LR_64x64` | `srvar_encoder` | 非空 | 16 (4×4) | LR_VAE mean | 方案 A 的变体：SRVAR encoder 进 KV，LR_VAE 只覆盖 scale[0] target |
| `LR` (256) | `srvar_encoder` | 空 | 256 (16×16) | SOS 预测 | **沿用旧 var 行为**：KV 长但信息量与 LR_64x64 相同 |
| `LR` (256) | `lr_vae` | * | * | * | **非法**：LR_VAE 期望 64×64 输入，启动 assert 报错 |

启动时（`SRtrain.py`）严格 assert 上面这张表。

---

## 4. 与 `patch_nums` 的耦合

只在启用 LR_VAE 覆盖 `scale[0]` target 时有强约束：

- LR_VAE encode 出来的 latent 边长 = `lr_img_size / 16`（默认 4，对应 `LR_64x64`）。
- 此时 `patch_nums[0]` 必须等于该边长（默认必须是 4）。
- 启动时 assert：`assert lr_vae.encode_to_posterior_mean(lr_64).shape[-1] == patch_nums[0]`。

如果只是 `srvar_encoder` 走 KV，没有覆盖 target，`patch_nums[0]` 不受 LR 约束（可以保留 `1` 起步，最小尺度由 SOS 预测）。

---

## 5. CFG `cfg_uncond` buffer 长度

`SRVAR.cfg_uncond` 是 `nn.Parameter(B=1, tlen, C=Cvae)`，CFG dropout 时按位置切片替换 `kv_compact`。
`tlen` 必须 `>= low_len`：

- `low_len = 16`（`LR_64x64`）→ `tlen >= 16`
- `low_len = 256`（`LR`）→ `tlen >= 256`

`--tlen` 默认设大一些（例如 1024）即可同时兼容；启动时仍 assert `tlen >= low_len`。

---

## 6. 常用切换示例

```bash
# 默认 (方案 B + LR_64x64 + srvar_encoder)
LR_FOLDER=LR_64x64 LR_COND_SOURCE=srvar_encoder bash SRtrain.sh

# 沿用旧 var 行为 (LR_256 + srvar_encoder)
LR_FOLDER=LR LR_COND_SOURCE=srvar_encoder bash SRtrain.sh

# 方案 A (stage1 LR_VAE 接 cross-attn 与 scale[0] prior)
LR_FOLDER=LR_64x64 LR_COND_SOURCE=lr_vae \
  STAGE1_CKPT=/path/to/stage1.pth \
  SKIP_SCALE0_LOSS=True \
  PATCH_NUMS_STR="4 5 6 8 10 13 16" \
  bash SRtrain.sh
```

---

## 7. 易错点

1. 把 `LR` (256) 误填到 `LR_FOLDER` 时若 `LR_COND_SOURCE=lr_vae`，必然 assert 失败。
2. `PATCH_NUMS_STR` 与 stage1 / stage2 ckpt 不一致时，要么 VAE 重建会乱，要么 LR_VAE 输出尺寸对不齐 → 启动期都会 assert。
3. `same_shape=True`（旧 var 默认）会把 LR_64 强制 bicubic 拉到 256；新版必须 `same_shape=False`，让 LR 原样进入 LR_VAE 或 SRVAR encoder。
4. 数据增广翻转 / center_crop_arr 要在 `(lr, hr)` 上同步执行，crop 区域在 HR 尺度上 sample 后按比例换算到 LR。
