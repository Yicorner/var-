---
name: continuous-ar-head
description: Explain why var uses MAR-style DiffLoss instead of CrossEntropy, how DiffLoss is wired in, and the design decisions around target/condition/CFG/sampling.
---

# Continuous AR Head (MAR-style DiffLoss)

> 面向修改 `models/diffusion/diffloss.py` 与 `models/SRVAR.py` 的人。讲清楚 var 把离散 CE 头换成 MAR 风格 DiffLoss 之后的所有关键决策。

---

## 1. 为什么不再用 CrossEntropy

myvaex 的多尺度 VAE 已经从 VQ 改成了连续 Gaussian posterior，没有 codebook。
var 直接复用这个 VAE 时面临的问题：

- 没有 `vocab_size`、没有 `embedding` 表，CrossEntropy 没有目标分布。
- 简单换成 MSE 回归会让 token 输出退化到 posterior mean，丢失多模态分布信息（连续 latent 仍带噪声）。

**解决方案**：参考 MAR（NeurIPS 2024）的设计，**每个空间位置上跑一次小扩散** 来建模该位置的连续 latent 分布。

---

## 2. DiffLoss 接口（MAR 原版）

文件：`models/diffusion/diffloss.py`

```python
class DiffLoss(nn.Module):
    def __init__(self, target_channels, z_channels, width, depth, num_sampling_steps,
                 grad_checkpointing=False):
        # train_diffusion: full 1000-step IDDPM (cosine schedule, learned_range)
        # gen_diffusion:   spaced N-step (e.g. "100") for inference
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,   # epsilon + learned sigma
            z_channels=z_channels,
            num_res_blocks=depth,
        )

    def forward(self, target, z) -> scalar_loss:
        # target: [N, target_channels]
        # z:      [N, z_channels]
        t = torch.randint(0, num_timesteps, (N,))
        loss = train_diffusion.training_losses(self.net, target, t, dict(c=z))["loss"]
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0) -> [N, target_channels]:
        # ancestral sampling via gen_diffusion.p_sample_loop
        # CFG: 复制 batch 一半 cond / 一半 uncond，net.forward_with_cfg
        ...
```

`target_channels = Cvae`（在 var 默认 32）；`z_channels = embed_dim`（默认 1024）。
**所有 scale 的 token 共享同一个 `DiffLoss`**（因为它们的通道维都是 `Cvae`）。

---

## 3. target 的取法：`posterior.mode()`

MAR 原版 `target = posterior.sample() * 0.2325`。
var 这里 VAE 是冻结的，**直接每次 forward 重新 sample 会让 target 抖动**，导致训练发散。

**约定**：

- target = `posterior.mode()`（即 mean）。
- 不做全局 latent rescale（不需要 MAR 的 `0.2325`）。
- 在 `models/quant.py` 的 `ContinuousMultiScaleQuantizer` 上新增 `forward_with_targets()` 或在 `vqvae.py` 上新增 `img_to_ms_continuous_input(img)`，**强制走 `deterministic=True` 路径**取出 mean。

---

## 4. `diffloss_batch_mul`

MAR 训练时把每个样本沿 N 维 `repeat_interleave(diffusion_batch_mul, 0)`：

- DiffLoss 是个轻量 MLP，单次 forward 算力消耗远低于 backbone，但 MSE 估计方差较大。
- 重复 4 次后 `(target, z)` 形成 `4 * B*L` 个独立的 `(t, eps)` 采样。
- 在 var 里同样建议 `diffloss_batch_mul=4`，可在 `arg_util.py` 暴露为 CLI 参数。

注意：重复仅发生在 **DiffLoss 内部**，不影响 backbone 的 batch；不会增加显存的瓶颈项。

---

## 5. CFG（Classifier-Free Guidance）

两层：

### 5.1 训练时（backbone 侧）

继承 var 原有 CFG dropout：以 `cond_drop_rate`（默认 0.1）把整条 `low_f` 替换成可学习的 `cfg_uncond` buffer。
注意 `cfg_uncond` 的形状要按 `low_len`（4×4=16 或 16×16=256）分配，启动时加 assert。

### 5.2 推理时（DiffLoss 头侧）

`DiffLoss.sample(z, temperature, cfg)` 内部：

```python
if cfg != 1.0:
    noise = torch.cat([noise, noise], 0)           # cond / uncond 各一半
    z_in = torch.cat([z, uncond_z], 0)             # 这里 var 的 uncond_z = pool(cfg_uncond)
    sample_fn = self.net.forward_with_cfg(cfg_scale=cfg)
else:
    sample_fn = self.net.forward
```

var 推理时把 backbone 的 forward 复制两份（cond / uncond_KV）跑完，得到 cond/uncond 的 `z`，再喂给 `DiffLoss.sample`；与 MAR 完全对齐。

---

## 6. 训练 1000 步 / 推理 spaced

- `train_diffusion = create_diffusion("")` → 默认 1000 步，cosine betas，`learn_sigma=True`。
- `gen_diffusion = create_diffusion(num_sampling_steps)` → 由 `--diff_steps` 控制，默认 `"100"`。
- 训练 loss 是「随机 t 的 epsilon MSE」，与 backbone 一致；不要在训练里调用 `gen_diffusion`。
- 推理时 spaced 的 100 步对每个 token 走 ancestral sampling；最大 scale 16×16=256 个 token batched 跑，开销可控。

---

## 7. 与多尺度的耦合

MAR 是 **单尺度**（一张图一个 token 网格），var 是 **多尺度**：

- token 总数 `L = sum_i pn_i^2`，跨尺度共享同一个 `SimpleMLPAdaLN`；通道维都是 `Cvae`，可以共享。
- `lvl_embed`（`SRVAR.py` 里已有）加在 transformer 输入上，**已经把尺度信息编码到了 z 里**，所以 DiffLoss 本身不需要再喂 scale id。
- 推理时按 scale 顺序逐个 sample，与离散 VAR 推理结构相同。

---

## 8. 与原 `forward_diff_loss(z, target, f_predict)` 的差别

旧 SRVAR 里 `forward_diff_loss` 把 `f_hat_predict`（即 argmax + idxBl_to_fhat 出来的离散 reconstr）拼到 `c` 上当 condition，让 DiffLoss 学一个「f - f_hat_pred」的残差。

**新设计直接抛弃**这种做法：

- 旧路径强依赖 `idx → embedding → f_hat` 链，与连续 VAE 不兼容。
- 直接 `target = ms_h_target[si]` 比残差更稳定，模型只需学每个位置的连续分布，没有 self-referential dependency。

---

## 9. 推荐默认参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `target_channels` | `Cvae`（默认 32） | 由 VAE ckpt 决定 |
| `z_channels` | `embed_dim`（默认 1024） | 与 SRVAR backbone 一致 |
| `width` (`diffloss_w`) | 1024 | MAR 的 large 默认 |
| `depth` (`diffloss_d`) | 3 | MAR 的 large 默认 |
| `num_sampling_steps` (`diff_steps`) | 100 | 推理子序列长度 |
| `diffloss_batch_mul` | 4 | DiffLoss 内 N 维 repeat |
| `cond_drop_rate` | 0.1 | 训练 CFG dropout |
| `cfg_infer` | 1.0 | 推理 CFG scale（>1 增强条件） |
