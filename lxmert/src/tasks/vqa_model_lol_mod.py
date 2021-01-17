# coding=utf-8
# Copyright 2019 project LXRT.
import torch
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU



# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 60


class VQAModel(nn.Module):
    def __init__(self, num_answers, fn_type="softmax"):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        
        hid_dim = self.lxrt_encoder.dim
        print("Size of Hidden Dimension:",hid_dim)
        fc_dim = int(hid_dim)
        print("Size of Hidden Dimension:",fc_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        
        if fn_type=="tanh":
            self.fn =self.tanh
            print("FN: TANH")
        elif fn_type=="softmax":
            self.fn= self.softmax
            print("FN: SOFTMAX")
        else:
            self.fn = self.sigmoid
            print("FN: SIGMOID")
        
        # YN:AND/OR/NOT/NONE Type Predictor
        self.yn_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 4)
        )
        
        # AND FF
        self.and_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )
        
        # OR FF
        self.or_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )
        
        # NOT FF
        self.not_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )
        
        # NONE FF
        self.none_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )
        
        # Type Predictor
        self.type_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 3)
        )
        
        # YESNO feedforward
        self.yesno_fc = nn.Sequential(
            nn.Linear(6*fc_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )

        # NUMBER feedforward
        self.number_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )

        # OTHER feedforward
        self.other_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        )

        # Answering Heads
        self.logit_fc1 = nn.Sequential(
            nn.Linear(4*fc_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        
        
        # Answering Heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)


    def ncat_vecs(self,list_of_vectors):
        v1 = list_of_vectors[0]
        v2 = list_of_vectors[1]
        v3 = list_of_vectors[2]
        v4 = list_of_vectors[3]

#         v_cat = torch.cat((v1, v2, v3, v4
#                            v1+v2+v3+v4, 
#                            v1*(v2+v3), v2*(v1+v3), v3*(v1+v2),
#                            v1*v2*v3), 1)
        v_cat = torch.cat((v1, v2, v3, v4,
                           v1+v2+v3+v4, v1*v2*v3*v4), 1)
        return v_cat

    def cat_vecs(self,list_of_vectors):
        v1 = list_of_vectors[0]
        v2 = list_of_vectors[1]
        v3 = list_of_vectors[2]

        v_cat = torch.cat((v1, v2, v3, v1+v2+v3), 1)

        return v_cat

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size


        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        
        batch_size = feat.size()[0]
        
        x = self.lxrt_encoder(sent, (feat, pos))
        
        yntype_logit = self.yn_fc(x)
        yn_wts = self.sigmoid(yntype_logit)
        
        #For YN Features:
        x_and = self.and_fc(x)
        repeat_dim = x_and.size()[-1]
        x_and = x_and*yn_wts.select(dim=1,index=0).reshape([batch_size,1]).repeat([1,repeat_dim]) # AND FEATURE WTS
        x_or = self.or_fc(x)
        x_or = x_or*yn_wts.select(dim=1,index=1).reshape([batch_size,1]).repeat([1,repeat_dim]) # OR FEATURE WTS
        x_not = self.not_fc(x)
        x_not = x_not*yn_wts.select(dim=1,index=2).reshape([batch_size,1]).repeat([1,repeat_dim]) # NOT FEATURE WTS
        x_none = self.none_fc(x)
        x_none = x_none*yn_wts.select(dim=1,index=3).reshape([batch_size,1]).repeat([1,repeat_dim]) # NONE FEATURE WTS
        
        
        # For QuesType Features
        type_logit = self.type_fc(x)
        type_wts = self.softmax(type_logit)
        
        # b*a.select(dim=1,index=2).reshape([3,1]).repeat([1,10])
        # YES-NO BRANCH
        x_yn = self.ncat_vecs([x_and,x_or,x_not,x_none])
        x_yn = self.yesno_fc(x_yn)
        repeat_dim = x_yn.size()[-1]
        x_yn = x_yn*type_wts.select(dim=1,index=0).reshape([batch_size,1]).repeat([1,repeat_dim])
        
        # NUMBER BRANCH
        x_num = self.number_fc(x)
        x_num = x_num*type_wts.select(dim=1,index=1).reshape([batch_size,1]).repeat([1,repeat_dim])
        
        # OTHER BRANCH
        x_other = self.other_fc(x)
        x_other = x_other*type_wts.select(dim=1,index=2).reshape([batch_size,1]).repeat([1,repeat_dim])

        x_cat = self.cat_vecs([x_yn, x_num, x_other])
        # answering head
        logit = self.logit_fc(self.logit_fc1(x_cat))

        return logit,yntype_logit,type_logit
