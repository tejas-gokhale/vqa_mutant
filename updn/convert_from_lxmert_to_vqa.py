import sys
import csv
import json 
import operator
import random 
import numpy as np 
import itertools 
from tqdm import tqdm


LXSPLIT = 'train'
if LXSPLIT == 'train':
	SPLIT = 'train'
elif LXSPLIT == 'minival':
	SPLIT = 'test'

# original VQACP annotations
VQACP_ANNO_PATH = './data/vqacp_v2_' + SPLIT + '_annotations.json'
VQACP_QUES_PATH = './data/vqacp_v2_' + SPLIT + '_questions.json'

# VQACP+MUTANT annotations in LXMERT format 
MUTANT_LXMERT_PATH = './data/mutant_only_vqacp_v2/' + LXSPLIT + '.json'

# write path for VQACP+MUTANT
MUTANT_ANNO_PATH = './data/mutant_only_vqacp_v2/vqacp_v2_' + SPLIT + '_annotations.json'
MUTANT_QUES_PATH = './data/mutant_only_vqacp_v2/vqacp_v2_' + SPLIT + '_questions.json'


# 1. load LXMERT annotations:
with open(MUTANT_LXMERT_PATH, 'r') as f:
	mutant_samples = json.load(f) 

	# for sample in mutant_samples:
	# 	if sample["orig_img_id"] == sample["img_id"]:
	# 		print(sample["mutation"])


print("loaded LXMERT annotations")

# # 2. read VQACP samples
# with open(VQACP_ANNO_PATH, 'r') as f:
# 	anno = json.load(f)
# 	print(len(anno))
# 	anno_info = anno["info"]
# 	anno_license = anno["license"]
# 	anno_data_subtype = anno["data_subtype"]
# 	anno_data_type = anno["data_type"]

# with open(*VQACP_QUES_PATH, 'r') as f:
# 	ques = json.load(f)
# 	ques_info = ques["info"]
# 	ques_license = ques["license"]
# 	ques_data_subtype = ques["data_subtype"]
# 	ques_data_type = ques["data_type"]
# 	ques_task_type = ques["task_type"]
# print("loaded VQACP original annotations")



with open(MUTANT_ANNO_PATH, 'w') as h, open(MUTANT_QUES_PATH, 'w') as f:
	data = []
	# data["info"] = anno_info	
	# data["license"] = anno_license
	# data["data_subtype"] = anno_data_subtype
	# data["data_type"] = anno_data_type
	# data["annotations"] = []

	qdata = []
	# qdata["info"] = ques_info	
	# qdata["license"] = ques_license
	# qdata["data_subtype"] = ques_data_subtype
	# qdata["data_type"] = ques_data_type
	# qdata["annotations"] = []
	# qdata["task_type"] = ques_task_type
	# qdata["questions"] = []


	for i in tqdm(range(len(mutant_samples)), ascii=True):
		sample = mutant_samples[i]

		new = {}
		new["answer_type"] = sample["answer_type"]
		new["image_id"] = sample["img_id"]
		new["question_type"] = sample["question_type"]	
		new["question_id"] = sample["question_id"]


		new["answers"] = []
		answer_id = 1
		v_max = 0
		new["multiple_choice_answer"] = "can't say"
		for k, v in sample["label"].items():
			if v > v_max:
				new["multiple_choice_answer"] = k
				v_max = v

			new["answers"].append({"answer": k,
								   "answer_confidence": v,
								   "answer_id" : answer_id})
			answer_id += 1

		if len(new["answers"]) == 0:
			new["answers"].append({"answer": "can't say",
								   "answer_confidence": 1,
								   "answer_id" : answer_id})


		if LXSPLIT != 'minival':
			### ORIG DATA
			new["orig_image_id"] = sample["orig_img_id"]
			new["orig_question_id"] = sample["orig_question_id"]


			new["orig_answers"] = []
			answer_id = 1
			v_max = 0
			new["orig_multiple_choice_answer"] = "can't say"
			for k, v in sample["orig_label"].items():
				if v > v_max:
					new["orig_multiple_choice_answer"] = k
					v_max = v

				new["orig_answers"].append({"answer": k,
									   "answer_confidence": v,
									   "answer_id" : answer_id})
				answer_id += 1

			if len(new["answers"]) == 0:
				new["orig_answers"].append({"answer": "can't say",
									   "answer_confidence": 1,
									   "answer_id" : answer_id})
		data.append(new)

		qnew = {}
		### MUTANT DATA
		qnew["image_id"] = sample["img_id"]
		qnew["question_id"] = sample["question_id"]
		qnew["question"] = sample["sent"]

		if LXSPLIT != 'minival':
			### ORIG DATA
			qnew["orig_image_id"] = sample["orig_img_id"]
			qnew["orig_question_id"] = sample["orig_question_id"]
			qnew["orig_question"] = sample["orig_sent"]


		try:
			qnew["mutation"] = sample["mutation"]
		except:
			qnew["mutation"] = 0

		try:
			qnew["w"] = sample["w"]
		except:
			qnew["w"] = 0

		qdata.append(qnew)


	json.dump(data, h)
	json.dump(qdata, f)












