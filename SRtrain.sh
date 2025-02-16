torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12346 SRtrain.py  \
--bs=256 --ep=500 \
--depth=16  --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=./data/brats_256_t1_pair \
--pn="1M" \
--rope2d_normalized_by_hw=2 --rope2d_each_sa_layer=1 \
--checkpointing="full-block" \
--vae_ckpt="ckpt-150.pth" \
--Ct5=32 --tlen=1024 

# --pad_to_multiplier=128 --use_flex_attn=True 
