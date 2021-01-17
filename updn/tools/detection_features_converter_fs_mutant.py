"""
Reads in a tsv file with pre-trained bottom up attention features and
stores them in per-image files.
"""
from __future__ import print_function

import os
import sys
from os.path import join, exists

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import numpy as np


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
infile_train = '/scratch/tgokhale/mutant/mutant_imgfeat/train_obj36.tsv'
infile_val = '/scratch/tgokhale/mutant/mutant_imgfeat/valid_obj36.tsv'
h_trainval_file = 'data/trainval36.hdf5'
feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    output_dir = 'data/trainval_features_mutant/'
    if not exists(output_dir):
        os.mkdir(output_dir)

    n_images = 10000000
    count = 0

    with open(infile_train, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for i, item in enumerate(reader):
            # pbar.update(1)
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [('features', (boxes, -1), np.float32)]

            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            arr = item["features"]
            if arr.shape[1] != 2048:
                count += 1
                print(count, i)

            image_id = '_'.join(item['img_id'].split('_')[2:])           
            out_file = join(output_dir, str(image_id) + ".bin")            
            arr.tofile(out_file)

        print(count)

    with open(infile_val, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for i, item in enumerate(reader):
            # pbar.update(1)
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [('features', (boxes, -1), np.float32),]

            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            arr = item["features"]
            if arr.shape[1] != 2048:
                count += 1
                print(count, i)

            image_id = '_'.join(item['img_id'].split('_')[2:])            
            out_file = join(output_dir, str(image_id) + ".bin")           
            arr.tofile(out_file)

        print(count)
    print("done!")
