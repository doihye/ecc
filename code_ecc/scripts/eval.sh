#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py \
--cfg configs/vit_base/simmim_finetune__vit_base__img224.yaml \
--data-path {data_path} \
--batch-size 512 \
--output {eval_output_path} \
--eval \
--resume output/diffmim_100ep_n2_finetune/ckpt_epoch_99.pth \