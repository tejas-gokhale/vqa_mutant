# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn

import torchvision.models as tvmodels



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
        
#         self.vis_log_fc = nn.Sequential(
#             nn.Linear(hid_dim*2, hid_dim * 10),
#             GeLU(),
#             BertLayerNorm(hid_dim * 10, eps=1e-12),
#             nn.Linear(hid_dim * 10, 36*3*30)
#         )

        self.vis_log_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 10),
            GeLU(),
            BertLayerNorm(hid_dim * 10, eps=1e-12),
            nn.Linear(hid_dim * 10, 36*3*30)
        )
    
        self.logit_fc_f1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 4),
            GeLU(),
            BertLayerNorm(hid_dim * 4, eps=1e-12),
            nn.Linear(hid_dim * 4, num_answers)
        )
        
#         self.logit_fc_f1 = nn.Sequential(
#             nn.Linear(hid_dim*2, hid_dim * 4),
#             GeLU(),
#             BertLayerNorm(hid_dim * 4, eps=1e-12),
#             nn.Linear(hid_dim * 4, 512)
#         )
        
#         self.logit_fc_f2 = nn.Sequential(
#             nn.Linear(512*36, hid_dim * 4),
#             GeLU(),
#             BertLayerNorm(hid_dim * 4, eps=1e-12),
#             nn.Linear(hid_dim * 4, num_answers)
#         )
        
#         self.image_p_size = 34
        self.image_p_size = 83
        
        self.visual_position_embeddings = nn.Embedding(self.image_p_size, 256)
        self.visual_embedding_mapper = nn.Sequential(
            nn.Linear(1000+256, 1000),
            GeLU(),
            torch.nn.LayerNorm(1000, eps=1e-12),
            nn.Linear(1000, hid_dim)
        )
        
        self.dropout = nn.Dropout(self.lxrt_encoder.model.config.hidden_dropout_prob)

        
        self.resnet = tvmodels.resnet34(pretrained=True)
        lf_enc_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=12)
        self.lf_enc = nn.TransformerEncoder(lf_enc_layer, num_layers=3)
        
        
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
        
        
    def get_resnet_feats(self,img_inps):
        bz = img_inps.shape[0]
        p77 = img_inps.unfold(2,128,64).unfold(3,96,48)
        p55 = img_inps.unfold(2,192,64).unfold(3,128,72)
        p33 = img_inps.unfold(2,256,128).unfold(3,192,96)
        p77 = p77.permute(0,3,2,1,4,5)
        p33 = p33.permute(0,3,2,1,4,5)
        p55 = p55.permute(0,3,2,1,4,5)
        p77 = p77.reshape([bz*49,3,128,96])
        p33 = p33.reshape([bz*9,3,256,192])
        p55 = p55.reshape([bz*24,3,192,128])
        feats_11 = self.resnet(img_inps).view([bz,1,1000])
        feats_33 = self.resnet(p33).view([bz,9,1000])
        feats_55 = self.resnet(p55).view([bz,24,1000])
        feats_77 = self.resnet(p77).view([bz,49,1000])
        full_feats = torch.cat([feats_11,feats_33,feats_55,feats_77],dim=1)
#         full_feats = torch.cat([feats_11,feats_33,feats_55],dim=1)
        assert full_feats.shape[1]==self.image_p_size
        return full_feats

    def forward(self, feat, pos, sent, img_raw_feats):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        
        visual_position_ids = torch.arange(self.image_p_size, dtype=torch.long).to(img_raw_feats.device)
        visual_position_ids = visual_position_ids.unsqueeze(0).expand([img_raw_feats.shape[0],self.image_p_size])
        visual_feats = self.get_resnet_feats(img_raw_feats)
        vpos_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = torch.cat([visual_feats,vpos_embeddings],dim=-1)
        visual_embeddings = self.visual_embedding_mapper(visual_embeddings)

        
        (l,v),x = self.lxrt_encoder(sent, (feat, pos))
        # v.shape = bz, 36, 768
        bz=v.shape[0]
        
        combined_embs = torch.cat([x.unsqueeze(1),v,l,visual_embeddings],dim=1)
        
        combined_embs = self.dropout(combined_embs)
        
        combined_txn_embs = self.lf_enc(combined_embs)
        
        x = combined_txn_embs[:,0,]
        v = combined_txn_embs[:,1:37,]
        
        x = self.dropout(x)
        v = self.dropout(v)
        
        logit = self.logit_fc_f1(x)
        v_logits = self.vis_log_fc(v) 
        
#         v_x_hids = torch.cat([torch.cat([x.unsqueeze(1)]*36,dim=1),v],dim=-1)
        
#         v_logits = self.vis_log_fc(v_x_hids)
        
#         x_hids = self.logit_fc_f1(v_x_hids).view([bz,512*36])
        
#         logit = self.logit_fc_f2(x_hids)
        
        
#         logit = self.logit_fc(x)
#         v_logits = self.vis_log_fc(v)
#         v_logits = v_logits.view([bz,3,36,36,3])
#         print(v_logits.shape)
        
        return logit, v_logits


