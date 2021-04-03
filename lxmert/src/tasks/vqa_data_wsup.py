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

import os

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
DATA_ROOT = "/data/data/lxmert_data/"
VQA_DATA_ROOT = "/data/data/lxmert_data/vqa/vqa_orig/"
MSCOCO_IMGFEAT_ROOT = '/data/data/lxmert_data/mscoco_imgfeat/'


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
    def __init__(self, splits: str,folder:str=None,nops:int=None):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in tqdm(self.splits,ascii=True,desc="Loading splits"):
            self.data.extend(json.load(open(VQA_DATA_ROOT+"/%s.json"%(split))))
        print("Data folder:%s"%folder,flush=True)
        print("Loading from %d data from split(s) %s."%(len(self.data), self.name),flush=True)
        
#         if nops is not None:
#             nops=int(nops)
#             if nops>1:
#                 print("Filtering with Nops:"+str(nops),flush=True)
#                 filtered = []
#                 for x in self.data:
#                     if x.get('n',1)> nops:
#                         filtered.append(x)
#                 self.data=filtered

        # Convert list to dict (for evaluation)
        self.id2datum = {
            ix : datum
            for ix,datum in tqdm(enumerate(self.data),ascii=True,desc="Converting List to Dict")
        }

        # Answers
        # if "vqacpv2" in folder:
        #     self.ans2label = json.load(open("/data/datasets/vqa_mutant/data/vqa/vqacpv2/trainval_ans2label.json"))
        #     self.label2ans = json.load(open("/data/datasets/vqa_mutant/data/vqa/vqacpv2/trainval_label2ans.json"))
        # else:
        #     self.ans2label = json.load(open("/data/datasets/vqa_mutant/data/vqa/trainval_ans2label.json"))
        #     self.label2ans = json.load(open("/data/datasets/vqa_mutant/data/vqa/trainval_label2ans.json"))
        self.label2ans = json.load(open(DATA_ROOT + "/merged_l2a.json"))
        self.ans2label = json.load(open(DATA_ROOT + "/merged_a2l.json"))

        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)
    
    
def updnlabels(c1,c2,ix):
    label = 0
    if c1[ix]> c2[ix]:
        label = 1
    elif c1[ix]==c2[ix]:
        label = 2
    return label

def create_spatial(cords):
    pair_wise_xyz_diffs = []
    pair_wise_xyz_labels =[]
    for oix1,c1 in enumerate(cords):
        for oix2,c2 in enumerate(cords):
            dxyz =  [c1[0]-c2[0],c1[1]-c2[1],c1[2]-c2[2]]
            pair_wise_xyz_diffs.append(dxyz)
            x_label = updnlabels(c1,c2,0)
            y_label = updnlabels(c1,c2,1)
            z_label = updnlabels(c1,c2,2)
            pair_wise_xyz_labels.append([x_label,y_label,z_label])
    return pair_wise_xyz_diffs, pair_wise_xyz_labels


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
        
        spatial_data = []
        spatial_data.extend(json.load(open(DATA_ROOT+"/vqa_depth/train2014_depth36.json")))
        spatial_data.extend(json.load(open(DATA_ROOT+"/vqa_depth/val2014_depth36.json")))
        spatial_data.extend(json.load(open(DATA_ROOT+"/vqa_depth/test2015_depth36.json")))
        
        if 'train' in dataset.splits:
#             img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mutant_imgfeat/train_obj36.tsv'), topk=topk))
#             img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mutant_imgfeat/valid_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))

        if 'valid' in dataset.splits:
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))
            
        if 'minival' in dataset.splits:
            # minival is 5K images in the intersection of MSCOCO valid and VG,
            # which is used in evaluating LXMERT pretraining performance.
            # It is saved as the top 5K features in val2014_obj36.tsv
            # img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mutant_imgfeat/valid_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))

        # if 'nominival' in dataset.splits:
            # nominival = mscoco val - minival
            # img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
#             img_data.extend(load_obj_tsv('/scratch/tgokhale/mutant_notcrowd/valid_obj36.tsv', topk=100))

            
        if 'test' in dataset.name:      # If dataset contains any test split
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/train2014_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
#             img_data.extend(load_obj_tsv('data/mscoco_imgfeat/test2015_obj36.tsv', topk=topk))

        # Convert img list to dict
        self.imgid2img = {}
        self.imgid2spa = {}
        
        for img_datum in spatial_data:
            self.imgid2spa[img_datum['img_id']] = img_datum
        
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        valid_imgs = []
        for datum in tqdm(self.raw_dataset.data,ascii=True,desc="Loading Image features"):
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
                valid_imgs.append(datum['img_id'])
        self.raw_dataset.data=self.data

        # Only keep images with loaded data 
        valid_imgs = set(valid_imgs)
        all_imgs = set(self.imgid2img)
        invalid_imgs = all_imgs - valid_imgs
        
        for unwanted_key in invalid_imgs:
            del self.imgid2img[unwanted_key]
        
        print("Use %d data in torch dataset" % (len(self.data)),flush=True)
        print(flush=True)

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

        bias = 0
        if "bias" in datum:
            bias= datum["bias"]
            
            
        spatial_datum = self.imgid2spa.get(str(img_id),None)
#         spatial_datum = None
        if spatial_datum is None:
            x = (boxes[:,(0)]+boxes[:,(2)])/2
            y = (boxes[:,(1)]+boxes[:,(3)])/2
            z =  np.zeros(36)
            cords = [ (a,b,c) for a,b,c in zip(x,y,z)]
            spatial_labels = create_spatial(cords)[1] #0: diffs, 1: labels
        else:
            spatial_labels = create_spatial(spatial_datum['3Dcoords_norm'])[1] #0: diffs, 1: labels

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
#                 if "-"!=ans and ans[0]=="-" and int(ans)<0 :
#                     ans='0'
                if ans not in self.raw_dataset.ans2label:
                    continue
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target, torch.tensor(spatial_labels)
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        atype_map={}
        acounts = {}
        
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.data[quesid]
            atype = datum["answer_type"]
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
        with open(path, 'w+') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


