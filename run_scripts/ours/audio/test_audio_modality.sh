#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc
conda env list
conda activate /opt/conda/envs/lavis
set -v
set -e
set -x

# train  video->audio
train_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/train/vicuna7b/adapter_in_adapter_audio_training.yaml
previous_modality_path=/15324359926/Multimodal/02_Only_adapter_LAVIS/LAVIS/lavis/output/xinstructblip/train/vicuna7b/video/video_training_frozen_and_adapter_no_relu_neckDim8_all_layer_learnable_scales/checkpoint_15000.pth
yq w -i ${train_yaml_path} model.pretrained_pc_qformer /${previous_modality_path}
output_dir_path=output/xinstructblip/train/vicuna7b/3D/adapter_in_adapter_audio_training_v2_ft16
yq w -i ${train_yaml_path} run.output_dir ${output_dir_path}

#python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ${train_yaml_path}  # train


####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/audiocaps_captioning_qa.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"


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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/audiocaps_captioning_test.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"


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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/audiocaps_captioning_val.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"


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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/clothoQA_captioning.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"


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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/clothov1_captioning.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"


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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/esc50_classification.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"

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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}

####################################################################################################################################################################################
test_yaml_path=/15324359926/Multimodal/hm_repo/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/audio/esc50_classification_completion.yaml
test_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
load_type="audio"
special_ckpt_path=/15324359926/Multimodal/03_adapter_in_adapter/LAVIS/lavis/${output_dir_path}/checkpoint_65000.pth
special_type="audio"

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

python  -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ${test_yaml_path}