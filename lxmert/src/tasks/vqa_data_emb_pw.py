# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import pickle
import os 

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
TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000

# The path to data and image features.
DATA_ROOT = '/data/datasets/vqa_mutant/data/'
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'


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
    def __init__(self, splits: str,folder: str,nops:int=None):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in tqdm(self.splits,ascii=True,desc="Loading splits"):
            self.data.extend(json.load(open("/data/datasets/vqa_mutant/data/vqa/%s/%s.json"%(folder,split))))
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
        
        self.ans2label = json.load(open("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/mutant_cp_ans2label.json"))
        self.label2ans = json.load(open("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/mutant_cp_label2ans.json"))
        

#             self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
#             self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
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
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mutant_imgfeat/train_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mutant_imgfeat/valid_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))

        if 'valid' in dataset.splits:
            print("VALID")
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))
            
        if 'minival' in dataset.splits:
            # minival is 5K images in the intersection of MSCOCO valid and VG,
            # which is used in evaluating LXMERT pretraining performance.
            # It is saved as the top 5K features in val2014_obj36.tsv
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
            img_data.extend(load_obj_tsv(os.path.join(DATA_ROOT, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))


        if 'test' in dataset.name:      # If dataset contains any test split
            # img_data.extend(load_obj_tsv('data/mscoco_imgfeat/train2014_obj36.tsv', topk=topk))
            # img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
            img_data.extend(load_obj_tsv('/data/datasets/vqa_mutant/data/mscoco_imgfeat/test2015_obj36.tsv', topk=topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        ### img_data IS DATA FROM THE TSV
        ### self.raw_dataset.data IS FROM THE JSON

        # Only kept the data with loaded image features
        self.data = []
        valid_imgs = []
        # count_orig = 0
        # count_mutant = 0
        # count_orig_not_key = 0
        # count_orig_not_in_tsv = 0
        for datum in self.raw_dataset.data:
            # print("datum.keys()", datum.keys())
            
            if 'minival' in dataset.splits:
                if datum['img_id'] in self.imgid2img:
                        self.data.append(datum)
                        valid_imgs.append(datum['img_id'])
            elif 'test' in dataset.splits:
                if datum['img_id'] in self.imgid2img:
                        self.data.append(datum)
                        valid_imgs.append(datum['img_id'])
            else:
                if 'img_id' in datum and 'orig_img_id' in datum:
                    if datum['img_id'] in self.imgid2img and datum['orig_img_id'] in self.imgid2img:
                        self.data.append(datum)
                        valid_imgs.append(datum['img_id'])
                        valid_imgs.append(datum['orig_img_id'])

            # if datum['orig_img_id'] in self.imgid2img:
            #         self.data.append(datum)
            #         valid_imgs.append(datum['orig_img_id'])
            #         count_orig += 1
            #     else:
            #         count_orig_not_in_tsv += 1
            # else:
            #     count_orig_not_key += 1 


        # print("count_mutant, count_orig, count_orig_not_in_tsv, count_orig_not_key", 
        #       count_mutant, count_orig, count_orig_not_in_tsv, count_orig_not_key, 
        #       flush=True)

        # self.raw_dataset.data=self.data
        
        # Only keep images with loaded data 
        valid_imgs = set(valid_imgs)
        all_imgs = set(self.imgid2img)
        invalid_imgs = all_imgs - valid_imgs
        
        for unwanted_key in invalid_imgs:
            del self.imgid2img[unwanted_key]
        
        print("Use %d data in torch dataset" % (len(self.data)),flush=True)
        print(flush=True)
        
        ans_embed = np.load("/home/tgokhale/code/vqa_mutant/mutant/answer_embs.npy") +1e-8
        ans_embed = torch.tensor(ans_embed)
        self.ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)
        
        self.nlp= spacy.load('en_core_web_lg')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        # print(item, "img_id", img_id, flush=True)

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

        target = torch.zeros(self.raw_dataset.num_answers)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            
            for ans, score in label.items():
                if ans not in self.raw_dataset.ans2label:
                    continue
                target[self.raw_dataset.ans2label[ans]] = score
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


            #### ORIG 
            if 'orig_img_id' in datum.keys():
                orig_img_id = datum['orig_img_id']
                orig_ques = datum['orig_sent']

                # Get image info
                try:
                    orig_img_info = self.imgid2img[orig_img_id]
                except:
                    return ques_id, feats, boxes, ques, target, torch.tensor(top_ans_emb)


                orig_obj_num = orig_img_info['num_boxes']
                orig_feats = orig_img_info['features'].copy()
                orig_boxes = orig_img_info['boxes'].copy()
                assert orig_obj_num == len(orig_boxes) == len(orig_feats)

                # Normalize the boxes (to 0 ~ 1)
                orig_img_h, orig_img_w = orig_img_info['img_h'], orig_img_info['img_w']
                orig_boxes = orig_boxes.copy()
                orig_boxes[:, (0, 2)] /= orig_img_w
                orig_boxes[:, (1, 3)] /= orig_img_h
                np.testing.assert_array_less(orig_boxes, 1+1e-5)
                np.testing.assert_array_less(-orig_boxes, 0+1e-5)

                # Provide label (target)
                # if 'orig_label' in datum:
                orig_label = datum['orig_label']
                orig_target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in orig_label.items():
                    if ans not in self.raw_dataset.ans2label:
                        continue
                    orig_target[self.raw_dataset.ans2label[ans]] = score
                orig_top_ans = [[k,v] for k, v in sorted(orig_label.items(), key=lambda item: item[1],reverse=True)]
                orig_top_ans_emb=None
                for tup in orig_top_ans:
                    if tup[0] in self.raw_dataset.ans2label:
                        orig_top_ans_emb = self.ans_embed[self.raw_dataset.ans2label[tup[0]]]
                        break
                if orig_top_ans_emb is None and len(orig_top_ans)>0:
                    ans = orig_top_ans[0][0]
                    orig_top_ans_emb = torch.nn.functional.normalize(torch.tensor(self.nlp(ans).vector),dim=-1)
                else:
                    orig_top_ans_emb = torch.rand([300])
                
                return ques_id, feats, boxes, ques, target, torch.tensor(top_ans_emb), \
                       orig_feats, orig_boxes, orig_ques, orig_target, torch.tensor(orig_top_ans_emb)
            else:
                # print("orig_img_id MISSING!!!")
                return ques_id, feats, boxes, ques, target, torch.tensor(top_ans_emb), \
                       feats, boxes, ques, target, torch.tensor(top_ans_emb)
        else:
            return ques_id, feats, boxes, ques, target, torch.rand([300]), \
                   feats, boxes, ques, target, torch.rand([300]) 


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
#             print(ans,label,flush=True)
            if ans in label:
                score += label[ans]
                atype_map[atype] = atype_map.get(atype,0.) + label[ans]
                
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


