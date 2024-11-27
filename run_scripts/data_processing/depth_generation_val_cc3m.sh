#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env list
conda activate /opt/conda/envs/depth
cd /15324359926/Multimodal/hm_repo/omnidata/omnidata_tools/torch
python -m torch.distributed.run    --nproc_per_node=2  demo_ddp.py  \
       --task depth  \
       --img_path /15324359926/Multimodal/hm_repo/CC3M/val \
       --output_path /15324359926/Multimodal/hm_repo/CC3M_Depth/val \
       --data_file  /15324359926/Multimodal/hm_repo/CC3M_Split/CC3M_val_1_5k.json \
       --bad_data_file /15324359926/Multimodal/hm_repo/CC3M_Split/bad_image_val_1_5k.json
