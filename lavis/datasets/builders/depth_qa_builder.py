"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset, VideoQAInstructDataset
from lavis.datasets.datasets.music_avqa import MusicAVQAInstructDataset, MusicAVQADataset
from lavis.datasets.datasets.depth_vqa_dataset import DepthQAInstructDataset, DepthQAInstructValDataset
from lavis.datasets.datasets.depth_vqa_dataset import DepthCaptionInstructDataset, DepthCaptionEvalDataset
from lavis.datasets.datasets.depth_vqa_dataset import DepthCaptionandQAInstructDataset, DepthCaptionandQAInstructValDataset
from lavis.datasets.datasets.depth_vqa_dataset import DepthCaptionandQAInstructValDataset_In_Domain

class DepthQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        # ans2label = self.config.build_info.annotations.get("ans2label")
        # if ans2label is None:
        #     raise ValueError("ans2label is not specified in build_info.")

        # ans2label = get_cache_path(ans2label.storage)

        # for split in datasets:
        #     datasets[split]._build_class_labels(ans2label)

        return datasets

@registry.register_builder("cc3m_depth_caption_instruct")
class CC3MDepthCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = DepthCaptionInstructDataset
    eval_dataset_cls = DepthCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/depth/cc3m_depth_instruct.yaml",
    }

@registry.register_builder("llava_depth_qa_instruct")
class DepthQAInstructBuilder(DepthQABuilder):
    train_dataset_cls = DepthQAInstructDataset
    eval_dataset_cls = DepthQAInstructValDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/depth/llava_depth_instruct.yaml",
    }

@registry.register_builder("fusion_depth_qa_instruct")
class DepthFusionInstructBuilder(DepthQABuilder):
    train_dataset_cls = DepthCaptionandQAInstructDataset
    eval_dataset_cls = DepthCaptionandQAInstructValDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/depth/fusion_depth_instruct.yaml",
    }

@registry.register_builder("fusion_depth_qa_instruct_in_domain")
class DepthFusionInstructBuilder_In_Domain(DepthQABuilder):
    train_dataset_cls = DepthCaptionandQAInstructDataset
    eval_dataset_cls = DepthCaptionandQAInstructValDataset_In_Domain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/depth/fusion_depth_instruct_in_domain.yaml",
    }


