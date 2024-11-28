import json
import os
import random
import tqdm
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.base_dataset import BaseDataset

TEMPLATES = [
"A short caption for the depth:",
"A short description of the depth:",
"A depth of",
"A depth that shows",
"Describe the depth briefly.",
"Write a description for the depth.",
"Provide a description of what is presented in the depth.",
"Briefly describe the content of the depth.",
"Can you briefly explain what you see in the depth?",
"Could you use a few words to describe what you perceive in the depth?",
"Please provide a short description of the depth.",
"Using language, provide a short account of the depth.",
"Use a few words to illustrate what is happening in the depth."
]

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        image_id = ann['image']
        vpath = os.path.join(self.vis_root, image_id)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class DepthQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.depth_root = '/path_to_your_data/LLaVA_Depth/train'
        print('train self.depth_root:{}'.format(self.depth_root))

    # def _build_class_labels(self, ans_path):
    #     ans2label = json.load(open(ans_path))

    #     self.class_labels = ans2label

    # def _get_answer_label(self, answer):
    #     if answer in self.class_labels:
    #         return self.class_labels[answer]
    #     else:
    #         return len(self.class_labels)

    def __getitem__(self, index):
        # assert (
        #     self.class_labels
        # ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]

        # vname = ann["video"]
        image_id = ann['image']
        conversations = ann['conversations']
        if len(conversations) > 1:
            question_list = []
            answer_list = []
            qa_list = []
            for conv in conversations:
                if conv['from'] == 'human':
                    question_list.append(conv['value'].replace('<image>', '').replace('\n',''))
                else:
                    answer_list.append(conv['value'])
            
            for idx, question in enumerate(question_list):
                qa_list.append({'question':question_list[idx], 'answer':answer_list[idx]})
                 
            conversation = random.choice(qa_list)
            question = conversation['question']
            answer = conversation['answer']
        else:
            question = conversation[0]['value']
            answer = conversation[1]['value']
        
        rgb_path = os.path.join(self.vis_root, image_id)
        depth_path = os.path.join(self.depth_root, image_id)
        # vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(rgb_path, depth_path)
        question = self.text_processor(question)
        answer = self.text_processor(answer)

        return {
            "depth": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": image_id,
            "instance_id": image_id,
        }

class DepthQAInstructDataset(DepthQADataset):
    def __getitem__(self, index):
        
        ann = self.annotation[index]

        # vname = ann["video"]
        image_id = ann['image']
        conversations = ann['conversations']
        if len(conversations) > 1:
            question_list = []
            answer_list = []
            qa_list = []
            for conv in conversations:
                if conv['from'] == 'human':
                    question_list.append(conv['value'].replace('<image>', '').replace('\n',''))
                else:
                    answer_list.append(conv['value'])
            
            for idx, question in enumerate(question_list):
                qa_list.append({'question':question_list[idx], 'answer':answer_list[idx]})
                 
            conversation = random.choice(qa_list)
            question = conversation['question']
            answer = conversation['answer']
        else:
            question = conversation[0]['value']
            answer = conversation[1]['value']
        
        rgb_path = os.path.join(self.vis_root, image_id)
        depth_path = os.path.join(self.depth_root, image_id.replace('.jpg', '.png'))
        # vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(rgb_path, depth_path)
        question = self.text_processor(question)
        answer = self.text_processor(answer)

        return {
            "video": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": image_id,
            "instance_id": image_id,
            ## add weight to use with vqa eval script
            "weight": [1.]
        }

class DepthQAInstructValDataset(DepthQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.depth_root = '/path_to_your_data/SUNRGBD'
        self.vis_root = '/path_to_your_data/SUNRGBD'
        self.class_labe = ['furniture_store', 'bedroom', 'classroom', 'living_room', 
                      'office', 'study_space', 'corridor', 'bathroom', 'conference_room', 
                      'dining_room', 'home_office', 'kitchen', 'discussion_area', 
                      'dining_area', 'rest_space', 'library', 'computer_room', 'lab', 
                      'lecture_theatre']
        self.classnames = ['furniture_store', 'bedroom', 'classroom', 'living_room', 
                      'office', 'study_space', 'corridor', 'bathroom', 'conference_room', 
                      'dining_room', 'home_office', 'kitchen', 'discussion_area', 
                      'dining_area', 'rest_space', 'library', 'computer_room', 'lab', 
                      'lecture_theatre']
        print('llava(SUN RGB-D) val self.depth_root:{}'.format(self.depth_root))
        print('llava(SUN RGB-D) val self.vis_root:{}'.format(self.vis_root))
        
    def __getitem__(self, index):
        
        ann = self.annotation[index]

        image_path = ann['image_path']
        label = ann['label']
        label_index = self.classnames.index(label) # return the index of label
        question = "{} What is the category of this scene? Choice one class from the class sets.".format(self.class_labe)
        
        rgb_path = os.path.join(self.vis_root, image_path)
        depth_path = os.path.join(self.vis_root, image_path.replace('.jpg', '.png'))
        # vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(rgb_path, depth_path)
        question = self.text_processor(question)
        answer = self.text_processor(label)

        return {
            "video": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": image_path,
            "instance_id": image_path,
            ## add weight to use with vqa eval script
            "weight": [1.],
            "label":label
        }
        
class DepthCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.depth_root = '/path_to_your_data/CC3M_Depth/train'
        print('caption train self.depth_root:{}'.format(self.depth_root))
        depth_set = set(os.listdir(self.depth_root))
        self.clear_annotations  = []
        print("clear data... ")
        for data in self.annotation:
            image_id = data["image_id"].split('/')[-1]
            depth_id = image_id.replace(".jpg", ".png")
            # depth_path = os.path.join(self.depth_root, depth_id)
            if depth_id in depth_set:
                self.clear_annotations.append(data)
            else:
                continue
        print('data cleared :{} !!'.format(len(self.clear_annotations)))
        # print('')
        self.annotation = self.clear_annotations
        
    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = ann["image_id"]
        image_id = image_path.split('/')[-1]
        depth_id = image_id.replace(".jpg", ".png")
        # video_path = os.path.join(self.vis_root, vname)
        # image_path = os.path.join(self.vis_root, image_id)
        depth_path = os.path.join(self.depth_root, depth_id)
        
        video = self.vis_processor(image_path, depth_path)
        
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": image_id,
        }


class DepthCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.depth_root = '/path_to_your_data/SUNRGBD'
        self.vis_root = '/path_to_your_data/SUNRGBD'
        self.class_labe = ['kitchen', 'office', 'bathroom', 'bedroom', 
                           'bookstore', 'living_room', 'study_space', 
                           'classroom', 'computer_room', 'lobby', 'home_office', 
                           'office_kitchen', 'playroom', 'reception_room', 'study', 
                           'dining_room']
        self.other_labe = ['computer_room', 'study', 'playroom', 'office_kitchen', 
                           'reception_room', 'lobby', 'study_space']      
        self.classnames = ['kitchen', 'office', 'bathroom', 'bedroom', 
                           'bookstore', 'living_room', 'classroom', 
                           'computer_room', 'home_office', 'dining_room',
                           'others']  
        print('caption(NYU) val self.depth_root:{}'.format(self.depth_root))
        print('caption(NYU) val self.vis_root:{}'.format(self.depth_root))
        # videos set. do not repeat videos in inference
        ## todo: make it deduplicated because creating annotation file makes 
        # seen = set()
        # self.annotation = [x for x in self.annotation if x["video"] not in seen and not seen.add(x["image_id"])]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image_id"]
        image_path = ann['image_path']
        label = ann['label']
        if label in self.class_labe and label in self.other_labe:
            label = 'others'
        
        depth_id = image_path.replace(".jpg", ".png")
        # video_path = os.path.join(self.vis_root, vname)
        image_path = os.path.join(self.vis_root, image_path)
        depth_path = os.path.join(self.depth_root, depth_id)
        
        video = self.vis_processor(image_path, depth_path)

        return {
            "video": video,
            "image_id": image_path,
            "instance_id": image_path,
            "label": label
        }


class DepthCaptionInstructDataset(DepthCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        # print(data)
        # assert 1==2
        return data

class DepthCaptionandQAInstructDataset(DepthQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.llava_depth_root = '/path_to_your_data/LLaVA_Depth/train'
        self.llava_image_root = '/path_to_your_data/data/coco/train2017'
        self.cc3m_depth_root = '/path_to_your_data/CC3M_Depth/train'
        self.cc3m_image_root = '/path_to_your_data/CC3M/train'
        print('train self.image_root:{} and {}'.format(self.llava_image_root, self.cc3m_image_root))
        print('train self.depth_root:{} and {}'.format(self.llava_depth_root, self.cc3m_depth_root))
        
        print('Using Cleared Dataset for training !!')
        
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        

        # vname = ann["video"]
        source = ann['source']
        
        if source == 'llava':
            image_id = ann['image']
            conversations = ann['conversations']
            if len(conversations) > 3:
                question_list = []
                answer_list = []
                qa_list = []
                for conv in conversations:
                    if conv['from'] == 'human':
                        question_list.append(conv['value'].replace('<image>', '').replace('\n',''))
                    else:
                        answer_list.append(conv['value'].replace('<image>', '').replace('\n',''))
                
                for idx, question in enumerate(question_list):
                    qa_list.append({'question':question_list[idx], 'answer':answer_list[idx]})
                    
                conversation = random.choice(qa_list)
                question = conversation['question']
                answer = conversation['answer']
            else:
                question = conversations[0]['value'].replace('<image>', '').replace('\n','')
                answer = conversations[1]['value'].replace('<image>', '').replace('\n','')
            
            rgb_path = os.path.join(self.llava_image_root, image_id)
            depth_path = os.path.join(self.llava_depth_root, image_id.replace('.jpg', '.png'))
            # vpath = os.path.join(self.vis_root, vname)

            frms = self.vis_processor(rgb_path, depth_path)
            question = self.text_processor(question)
            # answer = self.text_processor(answer)
            
        elif source == 'cc3m':
            image_id = ann["image_id"]
            question = random.choice(TEMPLATES)
            caption = ann["caption"]
            # image_id = image_path.split('/')[-1]
            depth_id = image_id.replace(".jpg", ".png")
            # video_path = os.path.join(self.vis_root, vname)
            image_path = os.path.join(self.cc3m_image_root, image_id)
            depth_path = os.path.join(self.cc3m_depth_root, depth_id)
            
            frms = self.vis_processor(image_path, depth_path)
            question = self.text_processor(question)
            # answer = self.text_processor(ann["caption"])
            answer = caption
        else:
            print(' Not Supported in this Mode !!')

        return {
            "depth": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": image_id,
            "instance_id": image_id,
            ## add weight to use with vqa eval script
            "weight": [1.]
        }

class DepthCaptionandQAInstructValDataset(DepthQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        dataset_name = ann_paths[0].split('/')[-1].split('-')[0]
        if 'NYU' in dataset_name:
            self.mode = 'nyu'
        elif 'SUN' in dataset_name:
            self.mode = 'sun'
        else:
            print('Dtatset not supported !!')
            self.mode = None
            
        assert self.mode != None, dataset_name
        
        self.depth_root = '/path_to_your_data/SUNRGBD'
        self.vis_root = '/path_to_your_data/SUNRGBD'
        
        if self.mode == 'sun':
            self.class_labe = ['furniture_store', 'bedroom', 'classroom', 'living_room', 
                        'office', 'study_space', 'corridor', 'bathroom', 'conference_room', 
                        'dining_room', 'home_office', 'kitchen', 'discussion_area', 
                        'dining_area', 'rest_space', 'library', 'computer_room', 'lab', 
                        'lecture_theatre']
            self.classnames = ['furniture_store', 'bedroom', 'classroom', 'living_room', 
                        'office', 'study_space', 'corridor', 'bathroom', 'conference_room', 
                        'dining_room', 'home_office', 'kitchen', 'discussion_area', 
                        'dining_area', 'rest_space', 'library', 'computer_room', 'lab', 
                        'lecture_theatre']
            
        elif self.mode == 'nyu':
            self.class_labe = ['kitchen', 'office', 'bathroom', 'bedroom', 
                            'bookstore', 'living_room', 'study_space', 
                            'classroom', 'computer_room', 'lobby', 'home_office', 
                            'office_kitchen', 'playroom', 'reception_room', 'study', 
                            'dining_room']
            self.other_labe = ['computer_room', 'study', 'playroom', 'office_kitchen', 
                            'reception_room', 'lobby', 'study_space']      
            self.classnames = ['kitchen', 'office', 'bathroom', 'bedroom', 
                            'bookstore', 'living_room', 'classroom',
                            'home_office', 'dining_room',
                            'others']  
            
        print('fusion ({}) val self.depth_root:{}'.format(self.mode, self.depth_root))
        print('fusion ({}) val self.vis_root:{}'.format(self.mode, self.vis_root))
        
    def __getitem__(self, index):
        
        ann = self.annotation[index]

        image_path = ann['image_path']
        label = ann['label']
        # label_index = self.classnames.index(label) # return the index of label
        if self.mode == 'nyu':
            if label in self.class_labe and label in self.other_labe:
                label = 'others'
                
        question = "{} What is the category of this scene? Choice one class from the class sets.".format(self.classnames)
        
        rgb_path = os.path.join(self.vis_root, image_path)
        depth_path = os.path.join(self.vis_root, image_path.replace('.jpg', '.png'))
        # vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(rgb_path, depth_path)
        question = self.text_processor(question)
        # answer = self.text_processor(label)
        answer = label

        return {
            "depth": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": image_path,
            "instance_id": image_path,
            ## add weight to use with vqa eval script
            "weight": [1.],
            "label":label
        }
          
class DepthCaptionandQAInstructValDataset_In_Domain(DepthQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print('Reading data from {} for validation'.format(ann_paths[0]))
        dataset_name = ann_paths[0].split('/')[-1].split('_')[0]
        if 'CC3M' in dataset_name:
            self.mode = 'cc3m'
        elif 'llava' in dataset_name:
            self.mode = 'llava'
        else:
            print('Dtatset not supported !!')
            self.mode = None
            
        assert self.mode != None, print(dataset_name)
        
        self.llava_depth_root = '/path_to_your_data/LLaVA_Depth/val'
        self.llava_image_root = '/path_to_your_data/LLaVA_Split/val'
        self.cc3m_depth_root = '/path_to_your_data/CC3M_Depth/val'
        self.cc3m_image_root = '/path_to_your_data/CC3M/val'
        print('val self.image_root:{} and {}'.format(self.llava_image_root, self.cc3m_image_root))
        print('val self.depth_root:{} and {}'.format(self.llava_depth_root, self.cc3m_depth_root))
        
        self.annotation = self.annotation[0]['data']
        # print(len(self.annotation))
        # print(self.annotation[0])
        # print(self.annotation[1])
        # assert 1==2
        if self.mode == 'cc3m':
            cc3m_depth_set = set(os.listdir(self.cc3m_depth_root))
            self.clear_annotations  = []
            print("clear data... ")
            for data in self.annotation:
                # print(data.keys())
                # print(data)
                # if data['source'] == 'cc3m':
                image_id = data["image_id"]
                depth_id = image_id.replace(".jpg", ".png")
                # depth_path = os.path.join(self.depth_root, depth_id)
                if depth_id in cc3m_depth_set:
                    self.clear_annotations.append(data)
                else:
                    continue
                # else:
                #     self.clear_annotations.append(data)
            print('data cleared :{} !!'.format(len(self.clear_annotations)))
        
            # print('')
            self.annotation = self.clear_annotations
        
    def __getitem__(self, index):
        
        ann = self.annotation[index]

        if self.mode == 'llava':
            image_id = ann['image_id']
            question = ann['question'].replace('<image>', '').replace('\n','')
            answer = ann['caption']
            
            rgb_path = os.path.join(self.llava_image_root, image_id)
            depth_path = os.path.join(self.llava_depth_root, image_id.replace('.jpg', '.png'))
            # vpath = os.path.join(self.vis_root, vname)

            frms = self.vis_processor(rgb_path, depth_path)
            question = self.text_processor(question)
            # answer = self.text_processor(answer)
            
            
        elif self.mode == 'cc3m':
            image_id = ann["image_id"]
            question = random.choice(TEMPLATES)
            # image_id = image_path.split('/')[-1]
            depth_id = image_id.replace(".jpg", ".png")
            # video_path = os.path.join(self.vis_root, vname)
            image_path = os.path.join(self.cc3m_image_root, image_id)
            depth_path = os.path.join(self.cc3m_depth_root, depth_id)
            
            frms = self.vis_processor(image_path, depth_path)
            question = self.text_processor(question)
            # answer = self.text_processor(ann["caption"])
            answer = ann["caption"]
        else:
            print(' Not Supported in this Mode !!')

        return {
            "depth": frms,
            "text_input": question,
            "answer": answer,
            "text_output": answer,
            "question_id": image_id,
            "instance_id": image_id,
            ## add weight to use with vqa eval script
            "weight": [1.],
            "image_id":image_id
        }