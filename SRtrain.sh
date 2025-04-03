export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12346 SRtrain.py  \
--bs=72 --ep=500 \
--tblr=0.0003 \
--alng=1e-3 --wpe=0.1 --data_path=../data/brats_256_t1_2021_pair_4x \
--pn="1M" \
--rope2d_normalized_by_hw=2 --rope2d_each_sa_layer=1 \
--enable_checkpointing="full-block" \
--vae_ckpt="vae_ch160v4096z32.pth" \
--Ct5=32 --vocab_size=4096 --tlen=1024 \
--tclip=5 \
--ada='0.9_0.96' \
--wpe=1 \
--wp=0.00000001 \
--fp16=1 \
--tini=-1 \
--val_and_saving_per_ep=1 \
--use_are_loss_weight=True \
--use_ref=True \
--use_diff=True
# --vae_ckpt="ckpt_256/ckpt-best.pth" \
# --pad_to_multiplier=128 --use_flex_attn=True 
# fp16 infity是2(bf16) var是1(fp16)

# python sendEmail.py