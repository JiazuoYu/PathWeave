#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env list
conda activate /opt/conda/envs/lavis
cp /path_to_your_data/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
cp /path_to_your_data/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0-models.jar /opt/conda/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice/lib
set -v
set -e
set -x

#################### pc    modality #####################
work_path=/path_to_your_data/03_adapter_in_adapter/depth2pc
cd /path_to_your_data/03_adapter_in_adapter/depth2pc
modelnet_cls=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/pc/modelnet40_classification_adapter.yaml
modelnet_ccompletion=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/pc/modelnet40_completion_adapter.yaml
cap3d_qa=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/pc/objaverse_qa_adapter.yaml
cap3d_caption=/path_to_your_data/03_adapter_in_adapter/depth2pc/lavis/projects/xinstruct_blip/eval/vicuna7b/pc/objaverse_captioning_adapter.yaml

# test pc modelnet_cls
test_dataset_name=modelnet_cls

current_modality=pc
# new_modalities='["depth"]'

current_dir_path=output/xinstructblip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training

current_modality_path=${work_path}/lavis/${current_dir_path}/checkpoint_65000.pth

yq w -i ${modelnet_cls} model.pretrained_${current_modality}_qformer ${current_modality_path}  # load current model
yq w -i ${modelnet_cls} model.load_ln_type_${current_modality} ${current_modality}
yq w -i ${modelnet_cls} model.load_qformer_type_${current_modality} ${current_modality}
yq w -i ${modelnet_cls} model.load_projection_type_${current_modality} ${current_modality}

# yq w -i ${msvd_qa} model.modalities ${new_modalities}

output_dir_path=output/xinstructblip/eval/vicuna7b/pc/${current_modality}/${test_dataset_name}
yq w -i ${modelnet_cls} run.output_dir ${output_dir_path}  # save output/xinstructblip/eval/vicuna7b/pc/${current_modality}/${test_modality}/${test_dataset_name}
 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${modelnet_cls}

#  pc modelnet_ccompletion
test_dataset_name=modelnet_ccompletion

current_modality=pc
# new_modalities='["depth"]'

current_dir_path=output/xinstructblip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training

current_modality_path=${work_path}/lavis/${current_dir_path}/checkpoint_65000.pth

yq w -i ${modelnet_ccompletion} model.pretrained_${current_modality}_qformer ${current_modality_path}
yq w -i ${modelnet_ccompletion} model.load_ln_type_${current_modality} ${current_modality}
yq w -i ${modelnet_ccompletion} model.load_qformer_type_${current_modality} ${current_modality}
yq w -i ${modelnet_ccompletion} model.load_projection_type_${current_modality} ${current_modality}

yq w -i ${modelnet_ccompletion} model.test_modality ${continue_modality}
# yq w -i ${msvd_qa} model.modalities ${new_modalities}

output_dir_path=output/xinstructblip/eval/vicuna7b/pc/${current_modality}/${test_dataset_name}
yq w -i ${modelnet_ccompletion} run.output_dir ${output_dir_path}
 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${modelnet_ccompletion}

#  pc cap3d_qa
test_dataset_name=cap3d_qa

current_modality=pc
# new_modalities='["depth"]'

current_dir_path=output/xinstructblip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training

current_modality_path=${work_path}/lavis/${current_dir_path}/checkpoint_65000.pth

yq w -i ${cap3d_qa} model.pretrained_${current_modality}_qformer ${current_modality_path}
yq w -i ${cap3d_qa} model.load_ln_type_${current_modality} ${current_modality}
yq w -i ${cap3d_qa} model.load_qformer_type_${current_modality} ${current_modality}
yq w -i ${cap3d_qa} model.load_projection_type_${current_modality} ${current_modality}

output_dir_path=output/xinstructblip/eval/vicuna7b/pc/${current_modality}/${test_dataset_name}
yq w -i ${cap3d_qa} run.output_dir ${output_dir_path}
 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${cap3d_qa}

# pc cap3d_caption
test_dataset_name=cap3d_caption

current_modality=pc

current_dir_path=output/xinstructblip/train/vicuna7b/adapter_in_adapter_pc_after_depth_training

current_modality_path=${work_path}/lavis/${current_dir_path}/checkpoint_65000.pth

yq w -i ${cap3d_caption} model.pretrained_${current_modality}_qformer ${current_modality_path}
yq w -i ${cap3d_caption} model.load_ln_type_${current_modality} ${current_modality}
yq w -i ${cap3d_caption} model.load_qformer_type_${current_modality} ${current_modality}
yq w -i ${cap3d_caption} model.load_projection_type_${current_modality} ${current_modality}

output_dir_path=output/xinstructblip/eval/vicuna7b/pc/${current_modality}/${test_dataset_name}

yq w -i ${cap3d_caption} run.output_dir ${output_dir_path}
python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path ${cap3d_caption}