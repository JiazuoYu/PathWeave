#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env list
conda activate /opt/conda/envs/lavis
cp /15324359926/Multimodal/hm_repo/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
cp /15324359926/Multimodal/hm_repo/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0-models.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
set -v
set -e
set -x

# train  depth->pc
cd /15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc
train_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training_in_domain.yaml
# previous_modality_path=/15324359926/Multimodal/0309-LAVIS-XInstructBLIP/trained_checkpoint/method_original/train_video_without_EncProj_without_any_method.pth
# yq w -i ${train_yaml_path} model.pretrained_pc_qformer /${previous_modality_path}
output_dir_path=output/xinstructblip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training
yq w -i ${train_yaml_path} run.output_dir ${output_dir_path}  # 保存到/15324359926/Multimodal/0309-LAVIS-XInstructBLIP/lavis/${output_dir_path}/checkpoint_15000.pth

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ${train_yaml_path}  # train
