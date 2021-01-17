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


def _create_entry(img_idx, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image_idx'       : img_idx,
        'question'    : question['question'],
        'answer'      : answer
    }
    return entry


def _load_dataset(dataroot, name, img_id2val, cp=False):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if cp:
      answer_path = os.path.join(dataroot, 'cp-cache', '%s_target.pkl' % name)
      name = "train" if name == "train" else "test"
      question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)
    else:
      question_path = os.path.join(
          dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)["questions"]
      answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)

    questions.sort(key=lambda x: x['question_id'])
    with open(answer_path, 'rb') as f:
      answers = cPickle.load(f)
    answers.sort(key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        if answer["labels"] is None:
            raise ValueError()
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        img_idx = None
        if img_id2val:
          img_idx = img_id2val[img_id]

        entries.append(_create_entry(img_idx, question, answer))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', cp=False,
                 use_hdf5=False, cache_image_features=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        if cp:
            ans2label_path = os.path.join(dataroot, 'cp-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cp-cache', 'trainval_label2ans.pkl')
        else:
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

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
                        fe = np.fromfile("data/trainval_features/" + str(img_id) + ".bin", np.float32)

                    image_to_fe[img_id] = torch.from_numpy(fe).view(36, 2048)
            self.image_to_fe = image_to_fe
            if use_hdf5:
                self.hf.close()
        else:
            self.image_to_fe = None

        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

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
        if self.image_to_fe is not None:
            features = self.image_to_fe[entry["image_id"]]
        elif self.use_hdf5:
            features = np.array(self.features[entry['image_idx']])
            features = torch.from_numpy(features).view(36, 2048)
        else:
            features = np.fromfile("data/trainval_features/" + str(entry["image_id"]) + ".data", np.float32)
            features = torch.from_numpy(features).view(36, 2048)

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if "bias" in entry:
            return features, question, target, entry["bias"]
        else:
            return features, question, target, 0

    def __len__(self):
        return len(self.entries)
