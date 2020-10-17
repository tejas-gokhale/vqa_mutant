# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model_emb import VQAModel
from tasks.vqa_data_emb_pw import VQADataset, VQATorchDataset, VQAEvaluator

from email.mime.text import MIMEText
from subprocess import Popen, PIPE
import socket

import numpy as np

import os
import psutil
process = psutil.Process(os.getpid())

import gc
import json

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, folder="/",nops=None) -> DataTuple:
    dset = VQADataset(splits,folder,nops=nops)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=0,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self,folder="/",load=True):
        # Datasets
        if load:
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True,folder=folder
            )
            if args.valid != "":
                self.valid_tuple = get_data_tuple(
                    args.valid, bs=128,
                    shuffle=False, drop_last=False, folder=folder,nops=args.nops
                )
            else:
                self.valid_tuple = None
        
        # Model
        self.ans2label = json.load(open("/data/datasets/vqa_mutant/data/vqa/mutant_l2a/mutant_cp_ans2label.json"))
        self.model = VQAModel(len(self.ans2label))

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            
        ans_embed = np.load("/home/tgokhale/code/vqa_mutant/mutant/answer_embs.npy") +1e-8
        ans_embed = torch.tensor(ans_embed).cuda()
        self.ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)
        self.embed_cache={}

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if load :
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
            # Output Directory
            self.output = args.output
            os.makedirs(self.output, exist_ok=True)
            
        self.cos = nn.CosineSimilarity()
    
    def get_answer_embs(self,bz):
        if bz in self.embed_cache:
            return self.embed_cache[bz]
        emb = torch.stack([self.ans_embed]*bz)
        self.embed_cache[bz]=emb
        return emb


    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        print("len(dset)", len(dset))
        print("len(loader)", len(loader))
        iter_wrapper = (lambda x: tqdm(x, total=len(loader),ascii=True)) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, batch in iter_wrapper(enumerate(loader)):
                # print(i, len(batch))

                ques_id, feats, boxes, sent, target, gold_embs, orig_feats, orig_boxes, orig_sent, orig_target, orig_gold_embs = batch

                self.model.train()
                self.optim.zero_grad()

                all_ans_embs = self.model.emb_proj(self.ans_embed)
                orig_all_ans_embs = self.model.emb_proj(self.ans_embed)
                cos = nn.CosineSimilarity()
                cos_dim2 = nn.CosineSimilarity(dim=2)

                ### MUTANT outputs
                feats, boxes, target, gold_emb = feats.cuda(), boxes.cuda(), target.cuda(), gold_embs.cuda()
                gold_emb = torch.nn.functional.normalize(gold_emb,dim=1)
                gen_embs,logits = self.model(feats, boxes, sent)
                all_ans_embs = torch.stack([all_ans_embs]*gen_embs.shape[0])
                gold_emb = self.model.emb_proj(gold_emb)

                ### MUTANT LOSSES
                positive_dist = cos(gen_embs, gold_emb) # shape b,k;b,k-> b
                gen_embs = torch.cat([gen_embs.unsqueeze(1)]*all_ans_embs.shape[1],dim=1)
                d_logit = cos_dim2(gen_embs,all_ans_embs)                
                num = torch.exp(positive_dist).squeeze(-1)
                den = torch.exp(d_logit).sum(-1)
                loss = -1 *torch.log(num/den)
                loss  = loss.mean() * d_logit.size(1)
                
                assert logits.dim() == target.dim() == 2
                acloss = self.bce_loss(logits, target)
                acloss = acloss * logits.size(1)
                
                mutant_loss = acloss+loss

                ### ORIG outputs
                orig_feats, orig_boxes, orig_target, orig_gold_emb = orig_feats.cuda(), orig_boxes.cuda(), orig_target.cuda(), orig_gold_embs.cuda()
                orig_gold_emb = torch.nn.functional.normalize(orig_gold_emb,dim=1)
                orig_gen_embs, orig_logits = self.model(orig_feats, orig_boxes, orig_sent)
                orig_all_ans_embs = torch.stack([orig_all_ans_embs]*orig_gen_embs.shape[0])
                orig_gold_emb = self.model.emb_proj(orig_gold_emb)           

                ### ORIG LOSSES
                orig_positive_dist = cos(orig_gen_embs, orig_gold_emb) # shape b,k;b,k-> b
                orig_gen_embs = torch.cat([orig_gen_embs.unsqueeze(1)]*orig_all_ans_embs.shape[1],dim=1)
                orig_d_logit = cos_dim2(orig_gen_embs, orig_all_ans_embs)                
                orig_num = torch.exp(orig_positive_dist).squeeze(-1)
                orig_den = torch.exp(orig_d_logit).sum(-1)
                orig_loss = -1 *torch.log(orig_num/orig_den)
                orig_loss  = orig_loss.mean() * orig_d_logit.size(1)
                
                assert orig_logits.dim() == orig_target.dim() == 2
                orig_acloss = self.bce_loss(orig_logits,orig_target)
                orig_acloss = orig_acloss * orig_logits.size(1)
                
                orig_loss = orig_acloss+orig_loss

                ### PW LOSS
                top_gen_embs, _ = gen_embs.max(1)
                top_orig_gen_embs, _ = orig_gen_embs.max(1)

                cos_gt = cos(gold_emb, orig_gold_emb)
                cos_pred = cos(top_gen_embs, top_orig_gen_embs)

                loss_pw = torch.abs(cos_gt - cos_pred).mean() * logits.size(1)


                ### TOTAL LOSS
                loss = mutant_loss + orig_loss + loss_pw

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logits.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans


                if i % 1000 == 0:
                    try:
                        iter_wrapper.set_description(
                            ", ".join(["LOSSES", 
                                       "mutant_loss", str(mutant_loss.item()),
                                       "acloss", str(acloss.item()),
                                       "orig_loss", str(orig_loss.item()), 
                                       "orig_acloss", str(orig_acloss.item()), 
                                       "loss_pw", str(loss_pw.item())
                                       ])
                            )
                        iter_wrapper.refresh()
                    except:
                        print("mutant_loss", mutant_loss.item(), "acloss", acloss.item(), 
                              "orig_loss", orig_loss.item(), "orig_acloss", orig_acloss.item(), 
                              "loss_pw", loss_pw.item())

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.) 

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")
        return best_valid

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in tqdm(enumerate(loader),ascii=True,desc="Evaluating"):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                embs,logits = self.model(feats, boxes, sent)
                # all_ans_embs = self.model.emb_proj(self.ans_embed)
                # all_ans_embs  = torch.stack([all_ans_embs]*embs.shape[0])
                # logit = torch.einsum("bj,bkj->bk",embs,all_ans_embs)
                score, label = logits.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":


    # Test or Train
    if args.test is not None:
            # Build Class
        vqa = VQA(folder=args.data,load=False)
            # Load VQA model weights
        # Note: It is different from loading LXMERT pre-trained weights.
        if args.load is not None:
            vqa.load(args.load)
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            test_tuple = get_data_tuple(args.test, bs=32,
                                   shuffle=False, drop_last=False,folder=args.data)
            quesid2ans = vqa.predict(test_tuple,dump=os.path.join('./', 'test_predict.json'))
            if "vqacpv2" in args.data:
                result = test_tuple.evaluator.evaluate(quesid2ans)
                print("Current Result:"+str(result))
                print("\n\n\n\n\n")
        elif int(args.nops)>1 and 'val' in args.test and not args.dump:
            for data in ["varlenlol","varlencwl"]:
                for nops in range(2,6): 
                    print("Data folder:",data, "Nops:",nops)
                    args.data=data
                    result = vqa.evaluate(
                        get_data_tuple('minival', bs=950,
                                       shuffle=False, drop_last=False,folder=args.data,nops=nops),
                        dump=os.path.join(args.output, 'yn_val_predict.json')
                    )
                    print("Current Result:"+str(result))
                    print("\n\n\n\n\n")
        elif 'val' in args.test and not args.dump:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
#             for data in ["orig","vqaorigyn","vqasinglelol","vqamultilol","vqalolval","vqacwlsingle","vqacwlmulti","vqacwlval","varlenlol","varlencwl","comma"]:
#             for data in ["flipped_lol","ordered_lol","flipped_cwl","ordered_cwl"]:
#                 print("Data folder:",data)
#                 args.data=data
            result = vqa.evaluate(
            get_data_tuple('minival', bs=950,
                                   shuffle=False, drop_last=False,folder=args.data,nops=args.nops),
                    dump=os.path.join('data/vqa/'+args.data, 'minival_predict.json')
            )
            print("Current Result:"+str(result))
            print("\n\n\n\n\n")
        elif 'val' in args.test and args.dump:
            vqa.predict(
                    get_data_tuple("minival", bs=950,
                                   shuffle=False, drop_last=False,folder=args.data,nops=args.nops),
                    dump=os.path.join('data/vqa/'+args.data, 'lxmert_val_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        # Build Class
        vqa = VQA(folder=args.data)
        # Load VQA model weights
        # Note: It is different from loading LXMERT pre-trained weights.
        if args.load is not None:
            vqa.load(args.load)
        
        print('Splits in Train data:', vqa.train_tuple.dataset.splits, flush=True)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits, flush=True)
            print(f'Process Memory : {process.memory_info().rss}')
            gc.collect()
            print(f'Process Memory : {process.memory_info().rss}')
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100),flush=True)
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
        
        #Send Mail when training ends
        hostname = socket.gethostname()
        msg = MIMEText(str(args)+"\n HostName:"+str(hostname)+"\n")
        msg["From"] = "tgokhale@asu.edu"
        msg["To"] = "tgokhale6@asu.edu"
        msg["Subject"] = "Job has ended."
        p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
        p.communicate(msg.as_bytes())

    
