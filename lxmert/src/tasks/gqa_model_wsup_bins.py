# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn


from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        
        self.vis_log_fc = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim * 10),
            GeLU(),
            BertLayerNorm(hid_dim * 10, eps=1e-12),
            nn.Linear(hid_dim * 10, 36*3*30)
        )
        
        self.logit_fc_f1 = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim * 4),
            GeLU(),
            BertLayerNorm(hid_dim * 4, eps=1e-12),
            nn.Linear(hid_dim * 4, 512)
        )
        
        self.logit_fc_f2 = nn.Sequential(
            nn.Linear(512*36, hid_dim * 4),
            GeLU(),
            BertLayerNorm(hid_dim * 4, eps=1e-12),
            nn.Linear(hid_dim * 4, num_answers)
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
        (l,v),x = self.lxrt_encoder(sent, (feat, pos))
        # v.shape = bz, 36, 768
        bz=v.shape[0]
        
        v_x_hids = torch.cat([torch.cat([x.unsqueeze(1)]*36,dim=1),v],dim=-1)
        
        v_logits = self.vis_log_fc(v_x_hids)
        
        x_hids = self.logit_fc_f1(v_x_hids).view([bz,512*36])
        
        logit = self.logit_fc_f2(x_hids)
        
        
#         logit = self.logit_fc(x)
#         v_logits = self.vis_log_fc(v)
#         v_logits = v_logits.view([bz,3,36,36,3])
#         print(v_logits.shape)
        
        return logit, v_logits


