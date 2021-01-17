from __future__ import print_function
import os
import json
import cPickle
from collections import Counter

import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ANS2LABEL_PATH = "data/mutant_only_vqacp_v2/mutant_cp_merge_ans2label.json"
LABEL2ANS_PATH = "data/mutant_only_vqacp_v2/mutant_cp_merge_label2ans.json"
answer_type_map = {"yes/no":[1,0,0,0], "number":[0,1,0,0], "other":[0,0,1,0], "color":[0,0,0,1]}
color_qtypes = ['what color is the', 'what color are the', 'what color', 
                'what color is', 'what is the color of the']

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img_idx, question, answer, anno):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'question_type': anno['question_type'],
        'image_id'    : question['image_id'],
        'image_idx'   : img_idx,
        'question'    : question['question'],
        'answer'      : answer, 
        'answer_type' : anno['answer_type']
    }
    return entry


def _load_dataset(dataroot, name, img_id2val, cp=False, tiny=False):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if cp:
        answer_path = os.path.join('data/mutant_both_vqacp_v2', 'cp-cache', '%s_target.pkl' % name)
        name = "train" if name == "train" else "test"
        question_path = os.path.join('data/mutant_both_vqacp_v2/', 'vqacp_v2_%s_questions.json' % name)
        anno_path = os.path.join('data/mutant_both_vqacp_v2/', 'vqacp_v2_%s_annotations.json' % name)
        with open(question_path) as f:
            questions = json.load(f)
        with open(anno_path) as f:
            annotations = json.load(f)
        with open(answer_path, 'rb') as f:
            answers = cPickle.load(f)       

    questions.sort(key=lambda x: x[u'question_id'])
    annotations.sort(key=lambda x: x[u'question_id'])
    answers.sort(key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    tiny_count = 0
    for question, answer, anno in zip(questions, answers, annotations):
        if answer["labels"] is None:
            raise ValueError()
        utils.assert_eq(question[u'question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        # print("question['image_id']", question['image_id'])
        img_idx = None
        if img_id2val:
            img_idx = img_id2val[img_id]

        entries.append(_create_entry(img_idx, question, answer, anno))

        tiny_count += 1
        if tiny and tiny_count > 100:
            break
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', cp=False,
                 use_hdf5=False, cache_image_features=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        if cp:
            ans2label_path = os.path.join('./data/mutant_both_vqacp_v2', 
                                'cp-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join('./data/mutant_both_vqacp_v2', 
                                'cp-cache', 'trainval_label2ans.pkl')
        else:
            ans2label_path = os.path.join(dataroot, 'cache', 
                                'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 
                                'trainval_label2ans.pkl')
        # self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        # self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.ans2label = json.load(open(ANS2LABEL_PATH))
        self.label2ans = json.load(open(LABEL2ANS_PATH))

        self.num_ans_candidates = len(self.ans2label)
        print("num_ans_candidates", self.num_ans_candidates)

        self.dictionary = dictionary
        self.use_hdf5 = use_hdf5

        if use_hdf5:
            h5_path = os.path.join(dataroot, 'trainval36.hdf5')
            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('image_features')

            with open("data/trainval36_imgid2idx.pkl", "rb") as f:
                imgid2idx = cPickle.load(f)
        else:
            imgid2idx = None

        self.entries = _load_dataset(dataroot, name, imgid2idx, cp=cp)

        if cache_image_features:
            image_to_fe = {}
            for entry in tqdm(self.entries, ncols=100, desc="caching-features"):
                img_id = entry["image_id"]
                if img_id not in image_to_fe:
                    if use_hdf5:
                        fe = np.array(self.features[imgid2idx[img_id]])
                    else:
                        
                        try:                          
                            img_id_str = '_'.join(img_id.split('_')[2:])
                            print("mutant", img_id, img_id_str)
                            fe = np.fromfile("data/trainval_features_mutant/" + str(img_id_str) + ".bin", np.float32)
                        except:                           
                            img_id_int = int(img_id.split('_')[-1])
                            print("original", img_id, img_id_int)
                            fe = np.fromfile("data/trainval_features/" + str(img_id_int) + ".bin", np.float32)
                    
                    ffff = torch.from_numpy(fe) 
                    image_to_fe[img_id] = ffff.view(36, 2048)

            self.image_to_fe = image_to_fe
            if use_hdf5:
                self.hf.close()
        else:
            self.image_to_fe = None

        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

        ans_embed = np.load("./data/mutant_both_vqacp_v2/answer_embs.npy") +1e-8
        ans_embed = torch.from_numpy(ans_embed).cuda()
        self.ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries, ncols=100, desc="tokenize"):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in tqdm(self.entries, ncols=100, desc="tensorize"):
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        whoa_count = 0
        if self.image_to_fe is not None:
            features = self.image_to_fe[entry["image_id"]]
        elif self.use_hdf5:
            features = np.array(self.features[entry['image_idx']])
            features = torch.from_numpy(features).view(36, 2048)
        else:
            img_id = entry["image_id"]
            try: 
                img_id_str = '_'.join(img_id.split('_')[2:])
                features = np.fromfile("data/trainval_features_mutant/" + str(img_id_str) + ".bin", np.float32)
            except:                           
                img_id_int = int(img_id.split('_')[-1])
                try:
                    features = np.fromfile("data/trainval_features/" + str(img_id_int) + ".bin", np.float32)
                except:
                    features = np.zeros((36, 2048), dtype=np.float32)
                    whoa_count += 1

            # features = np.fromfile("data/trainval_features/" + str(entry["image_id"]) + ".data", np.float32)
            features = torch.from_numpy(features).view(36, 2048)

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']

        if entry['answer_type'] == 'yes/no':
            answer_type = 0
        elif entry['answer_type'] == 'number':
            answer_type = 1
        else:
            answer_type = 2

        ### TYPETARGET
        typekey = entry['answer_type']
        qtypekey = entry['question_type']
        if qtypekey in color_qtypes:
            typekey = "color"

        ### ANSWERTYPEFEATS
        answertypefeats = answer_type_map[typekey]

        for ix,score in enumerate(answertypefeats):
            if score==1:
                typetarget=ix


        # answer_type = entry['answer_type']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
            try:
                _, max_idx = scores.max(0)
            except:
                max_idx = 0

            top_ans = labels[max_idx]
            try:
                top_ans_emb = self.ans_embed[int(top_ans)]
            except:
                top_ans_emb = self.ans_embed[int(top_ans[0])]

        else:
            top_ans_emb = self.ans_embed[0]


        if "bias" in entry:
            return features, question, typetarget, target, entry["bias"], torch.Tensor(answertypefeats), top_ans_emb
        else:
            return features, question, typetarget, target, 0, torch.Tensor(answertypefeats), top_ans_emb

    def __len__(self):
        return len(self.entries)
