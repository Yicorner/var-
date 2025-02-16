torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12346 SRtrain.py  \
--bs=256 --ep=500 \
--alng=1e-3 --wpe=0.1 --data_path=./data/brats_256_t1_pair \
--pn="1M" \
--rope2d_normalized_by_hw=2 --rope2d_each_sa_layer=1 \
--enable_checkpointing="full-block" \
--vae_ckpt="ckpt-150.pth" \
--Ct5=32 --tlen=1024 \
--tclip=5 \
--ada='0.9_0.96' \
--wpe=1 \
--wp=0.00000001 \
--fp16=1 \
--tini=-1 \
 # fp16 infity是2(bf16) var是1(fp16)