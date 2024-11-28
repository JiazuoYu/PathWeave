#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env list
conda activate /opt/conda/envs/depth
cd /path_to_your_data/omnidata/omnidata_tools/torch
python -m torch.distributed.run    --nproc_per_node=4  demo_ddp.py  \
       --task depth  \
       --img_path /path_to_your_data/CC3M/train \
       --output_path /path_to_your_data/CC3M_Depth/train \
       --data_file  /path_to_your_data/CC3M_Split/CC3M_train_25w_1.json \
       --bad_data_file /path_to_your_data/CC3M_Split/bad_image_25w_1.json
