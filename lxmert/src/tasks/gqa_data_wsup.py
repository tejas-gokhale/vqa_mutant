# coding=utf-8
# Copyleft 2019 project LXRT.

import json

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

ROOT_FOLDER = "/data/data/lxmert_data/"
VQA_DATA_ROOT = "/data/data/lxmert_data/vqa/vqa_orig/"
MSCOCO_IMGFEAT_ROOT = '/data/data/lxmert_data/mscoco_imgfeat/'

class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(ROOT_FOLDER+"/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))
        
        
        if "train" in splits:
            vqa_splits = "train,nominival"
            vqa_splits = vqa_splits.split(",")
            for split in tqdm(vqa_splits,ascii=True,desc="Loading splits VQA:"):
                self.data.extend(json.load(open(VQA_DATA_ROOT+"/%s.json"%(split))))
        

        # List to dict (for evaluation and others)
        self.id2datum = {
            int(datum['question_id']): datum
            for datum in self.data
        }

        # Answers
#         self.ans2label = json.load(open(ROOT_FOLDER+"/gqa/trainval_ans2label.json"))
#         self.label2ans = json.load(open(ROOT_FOLDER+"/gqa/trainval_label2ans.json"))
        self.ans2label = json.load(open(ROOT_FOLDER+"/merged_a2l.json"))
        self.label2ans = json.load(open(ROOT_FOLDER+"/merged_l2a.json"))      
        
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = ROOT_FOLDER+"/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = ROOT_FOLDER+"/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


import math

def updnlabels(c1,c2,ix):
    label = 0
    if c1[ix]> c2[ix]:
        label = 1
    elif c1[ix]==c2[ix]:
        label = 2
    return label

def normalize_cords(cords):
    cords = np.array(cords)
    for ix in range(0,3):
        cords[:,ix] = (cords[:,ix]-cords[:,ix].min())/(cords[:,ix].max()-cords[:,ix].min())
    return cords

def create_spatial(cords):
    cords = normalize_cords(cords)
    pair_wise_xyz_diffs = []
    pair_wise_xyz_labels =[]
    for oix1,c1 in enumerate(cords):
        if math.isnan(c1[0]) or math.isnan(c1[1]) or math.isnan(c1[2]):
            return None
        for oix2,c2 in enumerate(cords):
            dxyz =  [float(c1[0]-c2[0]),float(c1[1]-c2[1]),float(c1[2]-c2[2])]
            pair_wise_xyz_diffs.append(dxyz)
            x_label = updnlabels(c1,c2,0)
            y_label = updnlabels(c1,c2,1)
            z_label = updnlabels(c1,c2,2)
            pair_wise_xyz_labels.append([x_label,y_label,z_label])
    return pair_wise_xyz_diffs, pair_wise_xyz_labels


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        spatial_data = []
        spatial_data.extend(json.load(open(ROOT_FOLDER+"/gqa_depth/gqa_testdev_depth36.json")))
        spatial_data.extend(json.load(open(ROOT_FOLDER+"/gqa_depth/vg_gqa_depth36.json")))
        spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/train2014_depth36.json")))
        spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/val2014_depth36.json")))
        spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/test2015_depth36.json")))
        

        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        
        if "train" in dataset.splits:
            img_data.extend(load_obj_tsv(os.path.join(ROOT_FOLDER, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(ROOT_FOLDER, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))
        

        self.imgid2img = {}
        self.imgid2spa = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum
        
        
        for img_datum in spatial_data:
            self.imgid2spa[img_datum['img_id']] = img_datum

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
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        
        spatial_datum = self.imgid2spa.get(str(img_id),None)
#         spatial_datum = None
        if spatial_datum is None:
            x = (boxes[:,(0)]+boxes[:,(2)])/2
            y = (boxes[:,(1)]+boxes[:,(3)])/2
            z =  np.zeros(36)
            cords = [ (a,b,c) for a,b,c in zip(x,y,z)]
            spatial_labels = create_spatial(cords)[1] #0: diffs, 1: labels
        else:
            spatial_labels = create_spatial(spatial_datum['3Dcoords_camera'])[1] #0: diffs, 1: labels
            if spatial_labels is None:
                x = (boxes[:,(0)]+boxes[:,(2)])/2
                y = (boxes[:,(1)]+boxes[:,(3)])/2
                z =  np.zeros(36)
                cords = [ (a,b,c) for a,b,c in zip(x,y,z)]
                spatial_labels = create_spatial(cords)[1]
                if spatial_labels is None:
                    spatial_labels=np.zeros([1296,3])

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
                    
#             print(ques_id, feats, boxes, ques, target, torch.tensor(spatial_labels))
            return int(ques_id), feats, boxes, ques, target, torch.tensor(spatial_labels)
        else:
#             print( ques_id, feats, boxes, ques)
            return int(ques_id), feats, boxes, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[int(quesid)]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


