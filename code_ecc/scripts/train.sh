#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 main_simmim.py \
--cfg configs/vit_base/simmim_pretrain__vit_base__img224__400ep.yaml \
--data-path {dataset} \
--batch-size 512 \
--output {output_path} \
--accumulation-steps 2 \
--resume {warmup_ckpt} \