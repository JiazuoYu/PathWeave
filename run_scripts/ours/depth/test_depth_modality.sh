#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /path_to_your_data/03_adapter_in_adapter/depth2pc
conda env list
conda activate /opt/conda/envs/lavis
cp /path_to_your_data/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
cp /path_to_your_data/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0-models.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
set -v
set -e
set -x

#################################################################out-domain##################################################################################################
test_yaml_path=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/depth/adapter_in_adapter_depth_after_audio_training_out_domain.yaml
test_ckpt_path=/path_to_your_data/03_adapter_in_adapter/audio2depth/lavis/output/xinstructblip/train/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain/checkpoint_35000.pth
load_type="depth"
special_ckpt_path=/path_to_your_data/03_adapter_in_adapter/audio2depth/lavis/output/xinstructblip/train/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain/checkpoint_35000.pth
special_type="depth"
lora_bottleneck=8
output_dir_path=output/xinstructblip/eval/vicuna7b/adapter_in_adapter_depth_after_audio_training_out_domain_nyu
test_data_set=/path_to_your_data/SUNRGBD/NYU-Depth-v2_val.json
# /path_to_your_data/SUNRGBD/SUN-RGBD_val.json
# /path_to_your_data/SUNRGBD/NYU-Depth-v2_val.json


yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_depth ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_depth ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_depth ${load_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_depth_qformer ${test_ckpt_path}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}

yq w -i ${test_yaml_path} datasets.fusion_depth_qa_instruct.build_info.annotations.val.storage ${test_data_set}

yq w -i ${test_yaml_path} model.lora_bottleneck ${lora_bottleneck}

yq w -i ${test_yaml_path} run.output_dir ${output_dir_path}

#python  -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${test_yaml_path}


output_dir_path=output/xinstructblip/eval/vicuna7b/adapter_in_adapter_depth_after_audio_training_out_domain_sun
test_data_set=/path_to_your_data/SUNRGBD/SUN-RGBD_val.json
# /path_to_your_data/SUNRGBD/SUN-RGBD_val.json
# /path_to_your_data/SUNRGBD/NYU-Depth-v2_val.json


yq w -i ${test_yaml_path} datasets.fusion_depth_qa_instruct.build_info.annotations.val.storage ${test_data_set}

yq w -i ${test_yaml_path} run.output_dir ${output_dir_path}

#python  -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${test_yaml_path}

#################################################################in-domain##############################################################################
test_yaml_path=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/depth/adatper_in_adapter_depth_after_audio_training_in_domain.yaml # 测试数据集对应的yaml配置文件
test_ckpt_path=/path_to_your_data/03_adapter_in_adapter/audio2depth/lavis/output/xinstructblip/train/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain/checkpoint_35000.pth # continual的模型
load_type="depth"
special_ckpt_path=/path_to_your_data/03_adapter_in_adapter/audio2depth/lavis/output/xinstructblip/train/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain/checkpoint_35000.pth  # special的模型
special_type="depth"
lora_bottleneck=8
output_dir_path=output/xinstructblip/eval/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain_cc3m
test_data_set=/path_to_your_data/CC3M_Split/CC3M_val_1_5k_coco.json
# /path_to_your_data/process_llava150k/llava_instruct_1dot5k_test_data_coco.json
# /path_to_your_data/CC3M_Split/CC3M_val_1_5k_coco.json


yq w -i ${test_yaml_path} model.load_ln_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_ln_type_depth ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_qformer_type_depth ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_image ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_pc ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_video ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_audio ${load_type}
yq w -i ${test_yaml_path} model.load_projection_type_depth ${load_type}
yq w -i ${test_yaml_path} model.pretrained_image_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_pc_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_video_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_audio_qformer ${test_ckpt_path}
yq w -i ${test_yaml_path} model.pretrained_depth_qformer ${test_ckpt_path}

yq w -i ${test_yaml_path} model.special_module_path ${special_ckpt_path}
yq w -i ${test_yaml_path} model.special_module_type ${special_type}

yq w -i ${test_yaml_path} datasets.fusion_depth_qa_instruct_in_domain.build_info.annotations.val.storage ${test_data_set}

yq w -i ${test_yaml_path} model.lora_bottleneck ${lora_bottleneck}
yq w -i ${test_yaml_path} run.output_dir ${output_dir_path}

 python  -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${test_yaml_path}


output_dir_path=output/xinstructblip/eval/vicuna7b/adapter_in_adapter_depth_after_audio_training_in_domain_llava
test_data_set=/path_to_your_data/process_llava150k/llava_instruct_1dot5k_test_data_coco.json

yq w -i ${test_yaml_path} datasets.fusion_depth_qa_instruct_in_domain.build_info.annotations.val.storage ${test_data_set}
yq w -i ${test_yaml_path} run.output_dir ${output_dir_path}

python  -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${test_yaml_path}