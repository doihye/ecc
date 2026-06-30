#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py \
--cfg configs/vit_base/simmim_finetune__vit_base__img224.yaml \
--data-path {dataset} \
--pretrained {pretrained_ckpt} \
--batch-size 512 \
--output {output_dir} \
--accum_iter 2 \