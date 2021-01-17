# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
from tqdm import tqdm

import spacy

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'

answer_type_map = {"yes/no":[1,0,0,0],"number":[0,1,0,0],"other":[0,0,1,0],"color":[0,0,0,1]}

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str,folder: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in tqdm(self.splits,ascii=True,desc="Loading splits"):
            self.data.extend(json.load(open("/data/datasets/vqa_mutant/data/vqa/%s/%s.json"%(folder,split))))
        print("Data folder:%s"%folder,flush=True)
        print("Loading from %d data from split(s) %s."%(len(self.data), self.name),flush=True)

        # Convert list to dict (for evaluation)
        self.id2datum = {
#             datum['question_id']: datum
            ix: datum
            for ix,datum in tqdm(enumerate(self.data),ascii=True,desc="Converting List to Dict")
        }

        # Answers
#         self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
#         self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        self.ans2label = json.load(open("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/mutant_cp_merge_ans2label.json"))
        self.label2ans = json.load(open("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/mutant_cp_merge_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
                
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        if 'train' in dataset.splits:
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mutant_imgfeat/train_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mutant_imgfeat/valid_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/train2014_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'valid' in dataset.splits:
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'minival' in dataset.splits:
            # minival is 5K images in the intersection of MSCOCO valid and VG,
            # which is used in evaluating LXMERT pretraining performance.
            # It is saved as the top 5K features in val2014_obj36.tsv
#             if topk is None:
#                 topk = 5000
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/train2014_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'nominival' in dataset.splits:
            # nominival = mscoco val - minival
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'test' in dataset.name:      # If dataset contains any test split
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/test2015_obj36.tsv', topk=topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in tqdm(self.raw_dataset.data,ascii=True,desc="Loading Image features"):
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        self.raw_dataset.data = self.data
        print("Use %d data in torch dataset" % (len(self.data)),flush=True)
        print(flush=True)
        
        ans_embed = np.load("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/answer_embs.npy") + 1e-8
        ans_embed = torch.tensor(ans_embed)
        self.ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)
        
        self.nlp= spacy.load('en_core_web_lg')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = item
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            typetarget = 0
            answertypefeats = answer_type_map[datum['answer_type']]

            ans_index=[]
            not_present=[]
            ans_labels = list(label.keys())
            for ans, score in label.items():
                if ans not in self.raw_dataset.ans2label:
                    not_present.append(ans)
                    continue
                target[self.raw_dataset.ans2label[ans]] = score
                ans_index.append(self.raw_dataset.ans2label[ans])
            
            for ix,score in enumerate(answertypefeats):
                if score==1:
                    typetarget=ix
            
            top_ans = [[k,v] for k, v in sorted(label.items(), key=lambda item: item[1],reverse=True)]
            top_ans_emb=None
            for tup in top_ans:
                if tup[0] in self.raw_dataset.ans2label:
                    top_ans_emb = self.ans_embed[self.raw_dataset.ans2label[tup[0]]]
                    break
            if top_ans_emb is None and len(top_ans)>0:
                ans = top_ans[0][0]
                top_ans_emb = torch.nn.functional.normalize(torch.tensor(self.nlp(ans).vector),dim=-1)
            else:
                top_ans_emb = torch.rand([300])
            
            return ques_id, feats, boxes, ques, typetarget, target, torch.tensor(top_ans_emb) , torch.tensor(answertypefeats)
        else:
            return ques_id, feats, boxes, ques, 0,  torch.zeros(self.raw_dataset.num_answers) , torch.tensor(answertypefeats)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset
        self.color_qtypes = ['what color is the', 'what color are the', 
					 'what color', 'what color is',
					 'what is the color of the']

    def evaluate(self, quesid2ans: dict):
        score = 0.
        atype_map={}
        acounts = {}
        
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.data[quesid]
            atype = datum["answer_type"]
            qtypekey=datum['question_type']
            if qtypekey in self.color_qtypes:
                atype="color"
            label = datum['label']
            acounts[atype] = acounts.get(atype,0)+1

            if ans in label:
                score += label[ans]
                atype_map[atype] = atype_map.get(atype,0.) + label[ans]
        
        print(atype_map.keys())     
        for k,v in acounts.items():
            print(f"AnswerType Split:{k}:{atype_map.get(k,0)/v}:{v}",flush=True)

        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


