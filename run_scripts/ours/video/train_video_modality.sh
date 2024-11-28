#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /path_to_your_data/depth2pc
conda env list
conda activate /opt/conda/envs/lavis
set -v
set -e
set -x


train_yaml_path=/path_to_your_data/depth2pc/lavis/projects/xinstruct_blip/train/vicuna7b/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer.yaml
output_dir_path=output/xinstructblip/train/vicuna7b/video/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer_learnable_scales
yq w -i ${train_yaml_path} run.output_dir ${output_dir_path}
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ${train_yaml_path}