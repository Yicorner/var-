---
name: architecture
description: Document SRVAR's training/inference data flow, tensor shapes, and the contract between SRVAR and the upstream continuous multi-scale VAE.
---

# SRVAR Architecture

> 面向需要修改 `models/SRVAR.py` / `SRtrainer.py` 的人。说清楚训练和推理时每一步张量的形状、来源和去向。

---

## 1. 训练时的数据流（默认方案 B）

```text
[输入]
  inp_B3HW_super:  [B, 3, 256, 256]   # HR
  inp_B3HW_low:    [B, 3, 64, 64]     # LR_64x64（或 256）
  ref_B3HW (opt):  同上

[VAE 侧（torch.no_grad）]
  f_enc = vae.quant_conv(vae.encoder(inp_HR))            # [B, C=Cvae, 16, 16]
  for si, pn in enumerate(patch_nums):                    # patch_nums e.g. (1,2,3,4,5,6,8,10,13,16) 或与 ckpt 一致
      rest = F.interpolate(f_rest, (pn,pn), 'area') if si<SN-1 else f_rest
      moments = vae.quantize.mean_logvar_conv(rest)
      mean = DiagGaussian(moments, deterministic=True).mode()   # [B, C, pn, pn]
      ms_h_target.append(mean)                            # target for DiffLoss at scale si
      h_up = quant_resi[si/(SN-1)](upsample(mean, (16,16)))
      f_hat_cum += h_up
      f_rest -= h_up

  ms_x_input = cat over si in 0..SN-2 of:
      F.interpolate(f_hat_cum_at_end_of_si, (pn_{si+1}, pn_{si+1}), 'area').view(B,C,-1).transpose(1,2)
  # ms_x_input shape: [B, L-1, C]，L = sum(pn^2)

[SRVAR 侧]
  low_f = encoder_path(LR)                                # [B, low_len, C]
                                                          # srvar_encoder + LR_256: low_len=256
                                                          # srvar_encoder + LR_64:  low_len=16
                                                          # lr_vae   + LR_64:  low_len=16
  kv_compact = low_norm(low_f.reshape(-1,C))              # [B*low_len, C]
  sos = cond_BD = low_proj_for_sos(kv_compact)            # [B, D]
  ca_kv = (low_proj_for_ca(kv_compact), cu_seqlens_k, max_seqlen_k)

  sos = sos.unsqueeze(1) + pos_start                      # [B, 1, D]
  x_BLC = cat([sos, word_embed(norm0_ve(ms_x_input))],1)  # [B, L, D]
  x_BLC = add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule)
  x_BLC -> SelfAttn(block-causal mask per scale) + CrossAttn(ca_kv) ...
  z = x_BLC                                                # [B, L, D]

[DiffLoss head]
  z_flat = z.reshape(B*L, D)
  target_flat = cat_BL(ms_h_target).reshape(B*L, C)
  z_flat = z_flat.repeat_interleave(diffloss_batch_mul, 0)
  target_flat = target_flat.repeat_interleave(diffloss_batch_mul, 0)
  loss = diffloss(target=target_flat, z=z_flat)            # scalar
```

`L = sum_i pn_i^2`。当 `patch_nums = (1,2,...,16)` 时 `L = 1+4+9+...+256 = 680`。

`B*L` 个 token 各自被一个共享 `SimpleMLPAdaLN` 跑一次扩散训练。

---

## 2. 推理时的数据流

`autoregressive_infer_cfg` 的循环骨架：

```text
low_f = encoder_path(LR)              # 同上
ca_kv = build_kv(low_f)
sos = pool(low_f) + pos_start
last_stage = sos.unsqueeze(1)         # [B, 1, D]
accu_BChw = zeros(B, Cvae, 16, 16)

for si, (t, h, w) in enumerate(scale_schedule):
    z = transformer_blocks_for_scale_si(last_stage, ca_kv)   # [B, h*w, D]
    z_flat = z.reshape(B*h*w, D)
    h_pred = diffloss.sample(z_flat, temperature, cfg)        # [B*h*w, C]
    h_BChw = h_pred.view(B, h, w, C).permute(0, 3, 1, 2)

    accu_BChw, next_input = vae.quantize.get_next_autoregressive_input(
        si, SN, accu_BChw, h_BChw)
    last_stage = word_embed(norm0_ve(next_input.view(B,C,-1).transpose(1,2)))

img = vae.fhat_to_img(accu_BChw)     # [B, 3, 256, 256] in [-1, 1]
```

`scale_schedule` 默认从 `dynamic_resolution_h_w["1M"]["scales"]` 取，但 **必须** 与 `vae.patch_nums` 长度一致；不一致就强制退回 `[(1, pn, pn) for pn in patch_nums]`。

---

## 3. 模块清单

| 模块 | 文件 | 说明 |
|---|---|---|
| `encoder + quant_conv` | `SRVAR.py` 构造，权重通过 `init_LREncoder` 从冻结 VAE 复制 | 仅在 `lr_cond_source='srvar_encoder'` 时存在 |
| `TextAttentivePool` | `SRVAR.py` | 把 `low_f` 池化成单 token cond `[B, D]` |
| `low_norm` / `low_proj_for_sos` / `low_proj_for_ca` | `SRVAR.py` | 把 `low_f` 分别投到 SOS 与 KV 的 `D` 维 |
| `word_embed` | `SRVAR.py` | `Linear(C, D)` 把 teacher forcing 输入投到 transformer dim |
| `norm0_ve` | `SRVAR.py` | `LayerNorm` 在 word_embed 之前 |
| `shared_ada_lin` | `SRVAR.py` | `Linear(D, 6D)`，给每个 block 提供 AdaLN 的 gamma/beta |
| `unregistered_blocks` / `block_chunks` | `SRVAR.py` | `CrossAttnBlock` 堆叠 |
| `pos_start` / `lvl_embed` | `SRVAR.py` | 位置编码：SOS 起点 + 每个 scale 的 level embed |
| `cfg_uncond` | `SRVAR.py` | CFG 用的 unconditional KV buffer |
| `diffloss` | `models/diffusion/diffloss.py` | `SimpleMLPAdaLN` per-token diffusion head |

---

## 4. 已经移除的旧路径（重要：不要回退）

- `self.head: Linear(D, vocab_size)` 与 CrossEntropyLoss。
- `vae.quantize.embedding(idx)` / `gumbel_softmax_with_rng` / `sample_with_top_k_top_p_` / beam search。
- `forward_diff_loss(z, target, f_predict)` 「预测残差 `f - f_hat_pred`」的合一签名，已回归 MAR 原版 `forward(target, z)`。
- `gt_idx_Bl = vae.img_to_idxBl(...)` 和 `idxBl_to_var_input(idx)`：连续版改成 `vae.img_to_ms_continuous_input(img)` 直接拿 `ms_h_target` 和 `ms_x_input`。

---

## 5. 约束

1. `patch_nums` 与 myvaex stage2 ckpt 严格一致；同时 `vae_ch=128/160`、`Cvae=32`、`quant_resi=0.5`、`share_quant_resi=4` 都要按 ckpt 设。
2. `scale_schedule[i] = (t, h, w)`，对 1:1 图像通常 `t=1, h=w=patch_nums[i]`。
3. `low_len` 与 `cfg_uncond` 的 `tlen` 必须 `≥ low_f.shape[1]`。
4. 训练时 `vae.eval() + requires_grad_(False)`；`lr_vae` 同样要求；构造时 assert。
5. DiffLoss 训练用 `train_diffusion`（默认 1000 步），推理用 `gen_diffusion`（默认 `--diff_steps=100`）。
