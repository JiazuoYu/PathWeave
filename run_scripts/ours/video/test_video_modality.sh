#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc
conda env list
conda activate /opt/conda/envs/lavis
set -v
set -e
set -x

# training  # yq: re-writing
train_yaml_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/projects/xinstruct_blip/train/vicuna7b/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer.yaml
output_dir_path=output/xinstructblip/train/vicuna7b/video/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer_learnable_scales
yq w -i ${train_yaml_path} run.output_dir ${output_dir_path}
#python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ${train_yaml_path} # for training


# testing msvd_qa ####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/video/msvd_qa.yaml # YAML configuration file corresponding to the test dataset
test_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth # continual model
load_type="video"

special_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth  # special model
special_type="video"


yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}

python -m torch.distributed.run --nproc_per_node=2  train.py --cfg-path ${test_yaml_path} # test path1
###################################################################################################################################################################################################


# msvd_caption ####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/video/msvd_captioning.yaml
test_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
load_type="video"

special_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
special_type="video"



yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path} # 测path1
##################################################################################################################################################################################################

# msrvtt_caption_val ####################################################################################################################################################################################

test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/video/msrvtt_captioning_val.yaml
test_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
load_type="video"

special_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
special_type="video"


# load_type
yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}

python -m torch.distributed.run --nproc_per_node=2  train.py --cfg-path ${test_yaml_path} #
##################################################################################################################################################################################################

# msrvtt_qa_val ####################################################################################################################################################################################

test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/video/msrvtt_qa_val.yaml
test_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
load_type="video"

special_ckpt_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/${output_dir_path}/checkpoint_15000.pth
special_type="video"
#
# load_type
yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path} #
##################################################################################################################################################################################################
