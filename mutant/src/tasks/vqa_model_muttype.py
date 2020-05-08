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
        
        # Type Predictor
        self.type_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 4)
        )
        
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
        
        # YESNO feedforward
        self.yesno_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
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

         # OTHER feedforward
        self.color_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim *2),
            GeLU(),
            BertLayerNorm(hid_dim *2, eps=1e-12), 
            nn.Linear(2*hid_dim, fc_dim),
            GeLU(),
            BertLayerNorm(fc_dim, eps=1e-12)
        ) 
        
        # Answering Heads
        self.logit_fc1 = nn.Sequential(
            nn.Linear(5*fc_dim, hid_dim * 2),
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


        v_cat = torch.cat((v1, v2, v3, v4, 
                           v1+v2+v3+v4, 
                           v1*(v2+v3+v4), v2*(v1+v3+v4), v3*(v1+v2+v4), v4*(v1+v2+v3),
                           v1*v2*v3*v4), 1)

        return v_cat

    def cat_vecs(self,list_of_vectors):
        v1 = list_of_vectors[0]
        v2 = list_of_vectors[1]
        v3 = list_of_vectors[2]
        v4 = list_of_vectors[3]

        v_cat = torch.cat((v1, v2, v3, v4, v1+v2+v3+v4), 1)

        return v_cat
    
    def one_hot(self,batch,depth):
        ones = torch.sparse.torch.eye(depth).cuda()
        return ones.index_select(0,batch).cuda()

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
        
        type_logit = self.type_fc(x)
        
        # Question Type
        type_wts = self.softmax(type_logit)
        
#         type_wts = self.one_hot(type_wts.argmax(dim=1),3)
        
        # b*a.select(dim=1,index=2).reshape([3,1]).repeat([1,10])
        # YES-NO BRANCH
        x_yn = self.yesno_fc(x)
        repeat_dim = x_yn.size()[-1]
        x_yn = x_yn*type_wts.select(dim=1,index=0).reshape([batch_size,1]).repeat([1,repeat_dim])
        
        # NUMBER BRANCH
        x_num = self.number_fc(x)
        x_num = x_num*type_wts.select(dim=1,index=1).reshape([batch_size,1]).repeat([1,repeat_dim])
        
        # OTHER BRANCH
        x_other = self.other_fc(x)
        x_other = x_other*type_wts.select(dim=1,index=2).reshape([batch_size,1]).repeat([1,repeat_dim])

        x_color = self.color_fc(x)
        x_color = x_color*type_wts.select(dim=1,index=3).reshape([batch_size,1]).repeat([1,repeat_dim])

        x_cat = self.cat_vecs([x_yn, x_num, x_other,x_color])
        # answering head
        logit = self.logit_fc(self.logit_fc1(x_cat))

        return logit,type_logit
