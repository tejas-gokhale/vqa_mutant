import json
import spacy
import numpy as np 

nlp= spacy.load('en_core_web_lg')

LABEL_PATH = 'data/mutant_only_vqacp_v2/mutant_cp_merge_label2ans.json'

with open(LABEL_PATH, 'r') as f:
	labels = json.load(f)

ans_embs = []
for lbl in labels:
	emb = nlp(lbl).vector
	ans_embs.append(emb)

ans_embs = np.array(ans_embs)
print(ans_embs.shape)

np.save('data/mutant_only_vqacp_v2/answer_embs.npy', ans_embs)
# np.save('/data/datasets/vqa_mutant/data/vqa/mutant_l2a/answer_embs.npy', ans_embs)
