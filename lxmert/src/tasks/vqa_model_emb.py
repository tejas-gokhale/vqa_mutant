# coding=utf-8
# Copyleft 2019 project LXRT.
import torch
import torch.nn as nn
import numpy as np

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 300)
        )
        
        self.logit_fc_ans = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        
        self.emb_proj = nn.Sequential(
            nn.Linear(300, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, 300)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
    

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        gen_embs = self.logit_fc(x)
        gen_embs = torch.nn.functional.normalize(gen_embs,dim=1)
        logits = self.logit_fc_ans(x)
        return gen_embs,logits


