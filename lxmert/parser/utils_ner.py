# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import jsonlines
import json
from tqdm import tqdm
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels,counts,orig_row):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.counts = counts
        self.orig_row = orig_row


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, counts):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.counts = counts

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)
    
    @classmethod
    def _read_jsonlines(cls,input_file):
        """ Reads a jsonlines with QANER format """
        with jsonlines.open(input_file) as nerf:
            nerdata = [line for line in nerf]
        return nerdata
    
    @classmethod
    def _read_json(cls,input_file):
        return json.load(open(input_file))
        
class QANerProcessor(DataProcessor):
    def get_train_examples(self, data_dir,info="question"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train",info)

    def get_dev_examples(self, data_dir,info="question"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.json")), "dev",info)

    def get_test_examples(self, data_dir,info="question"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test",info)

    def get_labels(self):
        return ["B-ANS","I-ANS","O"]

    def _create_examples(self,lines,set_type,info=None):
        examples = []
        for i,item in tqdm(enumerate(lines[0:200000]),"Reading JsonFile"):
            guid = item["question_id"]
                
            if info is not None and item["n"]!=info:
                continue
                
            text = item['sent']
            text = text.replace("?"," ?")
            
            ans = item["r_list"]
                
            label = generate_bio(text,ans)
            words = text.split(" ")
            answer_len = len(ans)
            counts = 1 if answer_len > 0 else 0
            examples.append(InputExample(guid=guid,words=words,labels=label,counts=counts,orig_row=item))
        return examples
    

def read_examples_from_file_qa(data_dir,mode,info=None):
    processor = QANerProcessor()
    mode_examples = {"train":processor.get_train_examples,"dev":processor.get_dev_examples,"test":processor.get_test_examples}
    examples = mode_examples[mode](data_dir,info)
    return examples
        
        
def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                 words=words,
                                                 labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
    return examples


def get_matched(answer_tokens,ind,words_tokens):
    matched = False
    curr_match = False
    curr_ans_len = -1
    for ans_i in range(len(answer_tokens)):
        curr_match = False
        curr_ans_len = len(answer_tokens[ans_i])
        for token_A,token_B in zip(answer_tokens[ans_i],words_tokens[ind:ind+curr_ans_len]):
            if token_A==token_B:
                curr_match=True
            else:
                curr_match=False
                break
        if curr_match:
            matched=True
            break
    return curr_match,curr_ans_len


def generate_bio(text,answers):
    words_tokens = text.lower().split(' ')
    answer_tokens = [ answer.lower().split(' ') for answer in answers]
    tagged_tokens = []
    ind = 0
    while len(tagged_tokens) <= len(words_tokens):
        matched,len_ans = get_matched(answer_tokens,ind,words_tokens)
        if matched:
            tagged_tokens.append('B-ANS')
            num_of_is = ['I-ANS']*(len_ans-1)
            tagged_tokens.extend(num_of_is)
            ind += len_ans
        else:
            ind+=1
            tagged_tokens.append('O')
    tagged_tokens.pop(-1)
    if len(words_tokens) != len(tagged_tokens):
        print(text,words_tokens,tagged_tokens,len(words_tokens)-len(tagged_tokens))
    return tagged_tokens

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        while len(label_ids)> max_seq_length:
            label_ids.pop(-1)
        while len(label_ids)< max_seq_length:
            label_ids.append(pad_token_label_id)
            
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,counts=example.counts))
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
#         return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        return ["O","I-ANS","B-ANS"]