#!/bin/bash
# SRVAR continuous-AR training entry. Three example invocations:
#
#   # 1) DEFAULT (Plan B + LR_64x64 + srvar_encoder)
#   bash SRtrain.sh
#
#   # 2) Legacy-compatible (LR_256 upsampled + srvar_encoder)
#   LR_FOLDER=LR LR_COND_SOURCE=srvar_encoder bash SRtrain.sh
#
#   # 3) Plan A (Stage1 LR_VAE -> 4x4 KV + scale[0] prior, skip scale[0] loss)
#   LR_COND_SOURCE=lr_vae STAGE1_CKPT=/path/to/stage1.pth SKIP_SCALE0_LOSS=True \
#     PATCH_NUMS_STR="4 5 6 8 10 13 16" bash SRtrain.sh
#
# All variables below can be overridden by the environment.
set -e

# -------- experiment metadata --------
EXP_NAME=${EXP_NAME:-${exp_name:-srvar_continuous_default}}
EXP_NOTE=${EXP_NOTE:-${exp_note:-"SRVAR continuous AR with MAR-style DiffLoss head"}}
BED=${BED:-${bed:-local_output}}
PORT=${PORT:-${port:-13334}}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -------- data --------
DATA_PATH=${DATA_PATH:-${data_path:-/home/why/qbh/dataset/brats_256_t2_2021_pair_png_with_ref}}
LR_FOLDER=${LR_FOLDER:-${lr_folder:-LR_64x64}}
HR_FOLDER=${HR_FOLDER:-${hr_folder:-HR}}
SAME_SHAPE=${SAME_SHAPE:-${same_shape:-False}}

# -------- multi-scale (must match stage2 ckpt) --------
PATCH_NUMS_STR=${PATCH_NUMS_STR:-${PATCH_NUMS:-${patch_nums:-"1 2 3 4 5 6 8 10 13 16"}}}
read -r -a PATCH_NUMS <<< "$PATCH_NUMS_STR"

# -------- VAE interface --------
VAE_CKPT=${VAE_CKPT:-${vae_ckpt:-"vae_ch160v4096z32.pth"}}
CVAE=${CVAE:-${Ct5:-32}}
VAE_CH=${VAE_CH:-${vae_ch:-128}}
QUANT_RESI=${QUANT_RESI:-${quant_resi:-0.5}}
SHARE_QUANT_RESI=${SHARE_QUANT_RESI:-${share_quant_resi:-4}}

# -------- Stage1 (optional) --------
STAGE1_CKPT=${STAGE1_CKPT:-${stage1_ckpt:-""}}
LR_COND_SOURCE=${LR_COND_SOURCE:-${lr_cond_source:-srvar_encoder}}
SKIP_SCALE0_LOSS=${SKIP_SCALE0_LOSS:-${skip_scale0_loss:-False}}

# -------- training --------
EP=${EP:-${ep:-50}}
BS=${BS:-${bs:-4}}
AC=${AC:-${ac:-1}}
LR=${LR:-${tblr:-3e-4}}
WD=${WD:-${twd:-0.05}}
WP=${WP:-${wp:-0}}
GRAD_CLIP=${GRAD_CLIP:-${tclip:-2.0}}
FP16=${FP16:-${fp16:-1}}
TLEN=${TLEN:-${tlen:-1024}}

# -------- DiffLoss head --------
DIFFLOSS_W=${DIFFLOSS_W:-${diffloss_w:-1024}}
DIFFLOSS_D=${DIFFLOSS_D:-${diffloss_d:-3}}
DIFF_STEPS=${DIFF_STEPS:-${diff_steps:-100}}
DIFFLOSS_BATCH_MUL=${DIFFLOSS_BATCH_MUL:-${diffloss_batch_mul:-4}}
SCALE_LOSS_WEIGHTING=${SCALE_LOSS_WEIGHTING:-${scale_loss_weighting:-token}}
SCALE0_QUERY_SOURCE=${SCALE0_QUERY_SOURCE:-${scale0_query_source:-sos}}

# -------- CFG --------
CFG=${CFG:-${cfg:-0.1}}              # training-time condition dropout rate
CFG_INFER=${CFG_INFER:-${cfg_infer:-1.0}}  # inference-time CFG scale

# -------- validation / reconstruction visualisation --------
VAL_AND_SAVING_PER_EP=${VAL_AND_SAVING_PER_EP:-${val_and_saving_per_ep:-2}}
RECON_SAVE_INTERVAL=${RECON_SAVE_INTERVAL:-${reconstruction_save_interval:-0}}
RECON_MAX_SAMPLES=${RECON_MAX_SAMPLES:-${reconstruction_max_samples:-4}}
RECON_DIR_NAME=${RECON_DIR_NAME:-${reconstruction_dir_name:-reconstruction_samples}}
EVAL_AR_MAX_BATCHES=${EVAL_AR_MAX_BATCHES:-${eval_ar_max_batches:-4}}

# -------- logging / diagnostics --------
TRAIN_LOG_POINTS_PER_EPOCH=${TRAIN_LOG_POINTS_PER_EPOCH:-${train_log_points_per_epoch:-32}}
DIAGNOSTICS_ENABLED=${DIAGNOSTICS_ENABLED:-${diagnostics_enabled:-True}}
DIAGNOSTICS_INTERVAL=${DIAGNOSTICS_INTERVAL:-${diagnostics_interval:-0}}
DIAGNOSTICS_DIR_NAME=${DIAGNOSTICS_DIR_NAME:-${diagnostics_dir_name:-diagnostics}}
DIAGNOSTICS_MAX_SAMPLES=${DIAGNOSTICS_MAX_SAMPLES:-${diagnostics_max_samples:-4}}
DIAGNOSTICS_SAMPLE_SCALE0=${DIAGNOSTICS_SAMPLE_SCALE0:-${diagnostics_sample_scale0:-True}}

# Compose stage1 path additions only when STAGE1_CKPT is non-empty.
STAGE1_ARGS=()
if [ -n "$STAGE1_CKPT" ]; then
  STAGE1_ARGS+=(
    --stage1_ckpt="$STAGE1_CKPT"
    --lr_cond_source="$LR_COND_SOURCE"
    --skip_scale0_loss="$SKIP_SCALE0_LOSS"
  )
else
  # When stage1 is off, still allow toggling lr_cond_source (must remain
  # 'srvar_encoder' when STAGE1_CKPT is empty; the trainer asserts this).
  STAGE1_ARGS+=(--lr_cond_source="$LR_COND_SOURCE")
fi

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="$PORT" SRtrain.py \
  --exp_name="$EXP_NAME" --exp_note="$EXP_NOTE" \
  --local_out_dir_path="$BED" \
  --data_path="$DATA_PATH" \
  --lr_folder="$LR_FOLDER" --hr_folder="$HR_FOLDER" \
  --same_shape="$SAME_SHAPE" \
  --patch_nums "${PATCH_NUMS[@]}" \
  --vae_ckpt="$VAE_CKPT" \
  --Ct5="$CVAE" --vae_ch="$VAE_CH" \
  --quant_resi="$QUANT_RESI" --share_quant_resi="$SHARE_QUANT_RESI" \
  --tlen="$TLEN" \
  --pn="1M" --rope2d_normalized_by_hw=2 --rope2d_each_sa_layer=1 \
  --enable_checkpointing="full-block" \
  --bs="$BS" --ac="$AC" --ep="$EP" --tblr="$LR" --twd="$WD" --wp="$WP" --tclip="$GRAD_CLIP" \
  --fp16="$FP16" --tini=-1 \
  --val_and_saving_per_ep="$VAL_AND_SAVING_PER_EP" \
  --cfg="$CFG" --cfg_infer="$CFG_INFER" \
  --diffloss_w="$DIFFLOSS_W" --diffloss_d="$DIFFLOSS_D" \
  --diff_steps="$DIFF_STEPS" --diffloss_batch_mul="$DIFFLOSS_BATCH_MUL" \
  --scale_loss_weighting="$SCALE_LOSS_WEIGHTING" \
  --scale0_query_source="$SCALE0_QUERY_SOURCE" \
  --save_reconstruction_images=True \
  --reconstruction_save_interval="$RECON_SAVE_INTERVAL" \
  --reconstruction_max_samples="$RECON_MAX_SAMPLES" \
  --reconstruction_dir_name="$RECON_DIR_NAME" \
  --eval_ar_max_batches="$EVAL_AR_MAX_BATCHES" \
  --train_log_points_per_epoch="$TRAIN_LOG_POINTS_PER_EPOCH" \
  --diagnostics_enabled="$DIAGNOSTICS_ENABLED" \
  --diagnostics_interval="$DIAGNOSTICS_INTERVAL" \
  --diagnostics_dir_name="$DIAGNOSTICS_DIR_NAME" \
  --diagnostics_max_samples="$DIAGNOSTICS_MAX_SAMPLES" \
  --diagnostics_sample_scale0="$DIAGNOSTICS_SAMPLE_SCALE0" \
  --use_ref=False \
  "${STAGE1_ARGS[@]}"

# python sendEmail.py
