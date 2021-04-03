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

import jsonlines
import marshal
import random
import numpy as np
from bisect import bisect_left 
import math


from os import listdir
from os.path import isfile, join
from PIL import Image

import torchvision.transforms as T

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)




# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

ROOT_FOLDER = "/data/data/lxmert_data/"
VQA_DATA_ROOT = "/data/data/lxmert_data/vqa/vqa_orig/"
MSCOCO_IMGFEAT_ROOT = '/data/data/lxmert_data/mscoco_imgfeat/'


img_prefix = {
    "train": "/data/data/lxmert_data/vqa_train2014/",
    "val" : "/data/data/lxmert_data/vqa_val2014/",
    "test" : "/data/data/lxmert_data/vqa_test2015/",
    "gqa" : "/data/data/lxmert_data/gqa_images/",
    # "sbu" : "/data/datasets/patches/sbuimages/"
}


img_rcnn_prefix = {
    "gqa" : "/data/data/lxmert_data/updnfeats/bottomup_feats_vg/",
    "coco": "/data/data/lxmert_data/updnfeats/bottomup_feats/"
}


bin_walls = np.logspace(-100,0,7,base=1.08)
walls = list(-1* bin_walls)
walls.reverse()
walls.extend(list(bin_walls))
  
def interval_label(a, x): 
    i = bisect_left(a, x) 
    if i: 
        return (i-1) 
    else: 
        return 0
    
def get_image_prefix(img):
    src=None
    
    img = img+".jpg"
    
    if "COCO" in img:
        src="coco"
    if src=="coco" and "train" in img:
        return img_prefix["train"] + img
    if src=="coco" and "val" in img:
        return img_prefix["val"] + img
    if src=="coco" and "test" in img:
        return img_prefix["test"] + img
    return img_prefix["gqa"] + img

def get_rcnn_prefix(img):
    if "COCO" in img:
        return img_rcnn_prefix["coco"] + str(img)
    else:
        return img_rcnn_prefix["gqa"] + str(img)

def get_patches(imgfile,img=None):
    if img is None:
        try:
            # print(imgfile)
            img = Image.open(imgfile).convert('RGB')
            # img = resize_f(img)
        except Exception as e:
            print(img,e)
            raise
    # img_tensor = TF.to_tensor(img)
    return img

def get_transform():
    transforms = []
    transforms.append(T.Resize([512,384]))
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

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
        
#         if "train" in splits:
#             self.data.extend(json.load(open(ROOT_FOLDER+"/gqa/vqa_train_subset.json")))
        
#         if "train" in splits:
#             vqa_splits = "train,nominival"
#             vqa_splits = vqa_splits.split(",")
#             for split in tqdm(vqa_splits,ascii=True,desc="Loading splits VQA:"):
#                 self.data.extend(json.load(open(VQA_DATA_ROOT+"/%s.json"%(split))))
        

        # List to dict (for evaluation and others)
                         
        self.id2datum = {
            int(datum['question_id']): datum
            for datum in self.data
        }

        # Answers
#         self.ans2label = json.load(open(ROOT_FOLDER+"/gqa/trainval_ans2label.json"))
#         self.label2ans = json.load(open(ROOT_FOLDER+"/gqa/trainval_label2ans.json"))
#         self.ans2label = json.load(open(ROOT_FOLDER+"/merged2_a2l.json"))
#         self.label2ans = json.load(open(ROOT_FOLDER+"/merged2_l2a.json"))      
        
    
        self.ans2label = {"yes":0,"no":1,"others":2}
        self.label2ans = ["yes","no","others"]
    
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
#     cords = normalize_cords(cords)
    pair_wise_xyz_diffs = []
    pair_wise_xyz_labels =[]
    for oix1,c1 in enumerate(cords):
        if math.isnan(c1[0]):
            print("NAN Found:",c1,flush=True)
            c1[0]=0.5
        if math.isnan(c1[1]):
            print("NAN Found:",c1,flush=True)
            c1[1]=0.5
        if math.isnan(c1[2]):
            print("NAN Found:",c1,flush=True)
            c1[2]=0.5
        for oix2,c2 in enumerate(cords):
            dxyz =  [float(c1[0]-c2[0]),float(c1[1]-c2[1]),float(c1[2]-c2[2])]
            pair_wise_xyz_diffs.append(dxyz)
            x_label = updnlabels(c1,c2,0)
            y_label = updnlabels(c1,c2,1)
            z_label = updnlabels(c1,c2,2)
            pair_wise_xyz_labels.append([x_label,y_label,z_label])
    return pair_wise_xyz_diffs, pair_wise_xyz_labels


def spatial_diff_exp(spatial_diff):
    for s in spatial_diff:
        x,y,z = s
        s.extend([x*y,y*z,z*x,(x+y+z)/3,x*y*z])
    return spatial_diff


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset, is_valid=False):
        super().__init__()
        self.raw_dataset = dataset
        self.is_valid=is_valid

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
#         img_data = []
#         spatial_data = []
#         spatial_data.extend(json.load(open(ROOT_FOLDER+"/gqa_depth/gqa_testdev_depth36_l.json.bins.json.15")))

        

#         if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
#             img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
#         else:
#             img_data.extend(gqa_buffer_loader.load_data('train', topk))
        
#         if "train" in dataset.splits:
#             img_data.extend(load_obj_tsv(os.path.join(ROOT_FOLDER, 'mscoco_imgfeat/train2014_obj36.tsv'), topk=topk))
#             img_data.extend(load_obj_tsv(os.path.join(ROOT_FOLDER, 'mscoco_imgfeat/val2014_obj36.tsv'), topk=topk))
            
#             spatial_data.extend(json.load(open(ROOT_FOLDER+"/gqa_depth/vg_gqa_depth36_l.json.bins.json.15")))
#             print("Loaded VG feats")
#             spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/train2014_depth36_l.json.bins.json.15")))
#             print("Loaded COCO Train feats")
#             spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/val2014_depth36_l.json.bins.json.15")))
#             print("Loaded COCO Val feats")
#             spatial_data.extend(json.load(open(ROOT_FOLDER+"/vqa_depth/test2015_depth36_l.json.bins.json.15")))
#             print("Loaded COCO feats")
        

        self.imgid2img = {}
        self.imgid2spa = {}
#         for img_datum in img_data:
#             self.imgid2img[img_datum['img_id']] = img_datum
        
        
#         for img_datum in tqdm(spatial_data,desc="Converting to Numpy"):
#             img_datum['bins'] = np.array(img_datum['bins'])
#             self.imgid2spa[img_datum['img_id']] = img_datum
            
#         del spatial_data

        # Only kept the data with loaded image features
#         self.data = []
#         valid_imgs = []
#         for datum in tqdm(self.raw_dataset.data,ascii=True,desc="Loading Image features"):
#             if datum['img_id'] in self.imgid2img:
#                 self.data.append(datum)
#                 valid_imgs.append(datum['img_id'])
#         self.raw_dataset.data=self.data
        self.data = self.raw_dataset.data

        # Only keep images with loaded data 
#         valid_imgs = set(valid_imgs)
#         all_imgs = set(self.imgid2img)
#         invalid_imgs = all_imgs - valid_imgs
        
#         for unwanted_key in invalid_imgs:
#             del self.imgid2img[unwanted_key]
        
        print("Use %d data in torch dataset" % (len(self.data)),flush=True)
        print(flush=True)
        
        self.transforms = get_transform()
#         self.feat_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
#         img_info = self.imgid2img[img_id]
        
        output_path = get_rcnn_prefix(img_id)
        img_info = json.load(open(output_path))
        
        obj_num = img_info['num_boxes']
        boxes = np.array(img_info['boxes'])#.copy()
        feats = np.array(img_info['features'])#.copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
#         boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        
        try:
            spatial_datum = json.load(open("/data/data/lxmert_data/depth_bins/" + str(img_id) + ".7"))
            spatial_diff =  json.load(open("/data/data/lxmert_data/depth_diffs/" + str(img_id) + ".7"))
        except:
            spatial_datum = None
            spatial_diff = None
                
                                      
        
#         spatial_datum = self.imgid2spa.get(str(img_id),None)
#         spatial_datum = None
        if spatial_datum is None:
            print("Image Spatial Not Found:",img_id,flush=True)
            x = (boxes[:,(0)]+boxes[:,(2)])/2
            y = (boxes[:,(1)]+boxes[:,(3)])/2
            z =  np.zeros(36)
            cords = [(a,b,c) for a,b,c in zip(x,y,z)]
            diff = create_spatial(cords)[0] #0: diffs, 1: labels
            spatial_labels = [] 
            for trip in diff:
                trip_labels = [interval_label(walls,x) for x in trip]
                spatial_labels.append(trip_labels)
#             self.imgid2spa[str(img_id)] = {'bins':np.array(spatial_labels)}
            spatial_diff = diff
        else:
            spatial_labels = spatial_datum['bins']
            spatial_diff = spatial_diff['diff']
#             spatial_labels = create_spatial(spatial_datum['3Dcoords_camera'])[1] #0: diffs, 1: labels
#             if spatial_labels is None:
#                 x = (boxes[:,(0)]+boxes[:,(2)])/2
#                 y = (boxes[:,(1)]+boxes[:,(3)])/2
#                 z =  np.zeros(36)
#                 cords = [ (a,b,c) for a,b,c in zip(x,y,z)]
#                 spatial_labels = create_spatial(cords)[1]
#                 if spatial_labels is None:
#                     spatial_labels=np.zeros([1296,3])


        # Read Image:
        
#         if img_id in self.feat_cache and self.is_valid:
#             img_raw_data = self.feat_cache[img_id]
#         else:
#             img_path = get_image_prefix(img_id)
#             img_raw_data =  self.transforms(get_patches(img_path,None))
#             if self.is_valid:
#                 self.feat_cache[img_id]=img_raw_data

        spatial_diff = spatial_diff_exp(spatial_diff)
        
        img_path = get_image_prefix(img_id)
        img_raw_data =  self.transforms(get_patches(img_path,None))

        # Create target
        if 'label' in datum:
            label = datum['nlabel']
            target = torch.zeros(3)
            target[label]=1.0
#             for ans, score in label.items():
#                 if ans in self.raw_dataset.ans2label:
#                     target[self.raw_dataset.ans2label[ans]] = score
                    
#             print(ques_id, feats, boxes, ques, target, torch.tensor(spatial_labels))
            return int(ques_id), torch.tensor(feats).float(), torch.tensor(boxes).float(), ques, torch.tensor(spatial_diff).float(), img_raw_data, target, torch.tensor(spatial_labels)
        else:
#             print( ques_id, feats, boxes, ques)
            return int(ques_id), torch.tensor(feats).float(), torch.tensor(boxes).float(), ques, torch.tensor(spatial_diff).float(), img_raw_data


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[int(quesid)]
            label = datum['nlabel']
            if ans == self.dataset.label2ans[label]:
                score += 1.0
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
                    'questionId': str(ques_id.item()),
                    'prediction': str(ans)
                })
            json.dump(result, f, indent=4, sort_keys=True)


            
if __name__ == "__main__":
    dset = GQADataset("train,valid")
    tset = GQATorchDataset(dset)
    c=0
    for ix,row in tqdm(enumerate(tset)):
#         print(row)
#         if ix>5:
#             break
        c+=1
    print(c)
    
    dset = GQADataset("testdev")
    tset = GQATorchDataset(dset)
    c=0
    for ix,row in tqdm(enumerate(tset)):
#         print(row)
#         if ix>5:
#             break
        c+=1
    print(c)