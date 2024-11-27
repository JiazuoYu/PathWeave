#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc
conda env list
conda activate /opt/conda/envs/lavis
set -v
set -e
set -x


train_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/train/vicuna7b/adapter_in_adapter_audio_training.yaml
previous_modality_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/output/xinstructblip/train/vicuna7b/video/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer_learnable_scales/checkpoint_15000.pth
yq w -i ${train_yaml_path} model.pretrained_pc_qformer /${previous_modality_path}
output_dir_path=output/xinstructblip/train/vicuna7b/3D/adapter_in_adapter_audio_training_v2_ft16
yq w -i ${train_yaml_path} run.output_dir ${output_dir_path}

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ${train_yaml_path}  # train