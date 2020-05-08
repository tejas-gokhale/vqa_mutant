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
from tasks.vqa_model_muttype import VQAModel
from tasks.vqa_data_mutant_type import VQADataset, VQATorchDataset, VQAEvaluator

from email.mime.text import MIMEText
from subprocess import Popen, PIPE
import socket
import json

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, folder="/") -> DataTuple:
    dset = VQADataset(splits,folder)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
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
                    shuffle=False, drop_last=False, folder=folder
                )
            else:
                self.valid_tuple = None
        
        # Model
#         self.model = VQAModel(self.train_tuple.dataset.num_answers)
        self.model = VQAModel(len(self.train_tuple.dataset.label2ans),fn_type=args.fn_type)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model, label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            
        # Load IndexList of Answer to Type Map
        self.indexlist = json.load(open("data/vqa/mutant_indexlist.json"))

        print("Length of Masks",len(self.indexlist),flush=True)

        indextensor = torch.cuda.LongTensor(self.indexlist)
        self.mask0 = torch.eq(indextensor,0).float()
        self.mask1 = torch.eq(indextensor,1).float()
        self.mask2 = torch.eq(indextensor,2).float()
        self.mask3 = torch.eq(indextensor,3).float()
        
        self.mask_cache = {}
        
        # Loss and Optimizer
        
        self.logsoftmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.bceloss = nn.BCELoss()
        self.nllloss = nn.NLLLoss()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
                
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

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader),ascii=True)) if args.tqdm else (lambda x: x)

        total_steps = len(loader)
        eval_every =  int(0.2 * total_steps)

        best_valid = 0.
        best_i = 0
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, typetarget, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target, typetarget = feats.cuda(), boxes.cuda(), target.cuda(), typetarget.cuda()
                logit,type_logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                
                logit = self.sigmoid(logit)
                type_logit_soft = self.softmax(type_logit)
                type_logit = self.logsoftmax(type_logit)
                
                logit = self.calculatelogits(logit,type_logit_soft,"train")
                
                loss = self.bceloss(logit, target)
                loss = loss * logit.size(1)
                
                loss_type = self.nllloss(type_logit,typetarget)
                loss_type = loss_type* type_logit.size(1)
                
                loss = 0.9*loss + 0.1*loss_type

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                if ((i+1)%eval_every == 0) and self.valid_tuple is not None:
                    log_str = "\nEpoch %d, Step %d: Train %0.2f\n" % (epoch,i, evaluator.evaluate(quesid2ans) * 100.)

                    if self.valid_tuple is not None:  # Do Validation
                        valid_score = self.evaluate(eval_tuple)
                        if valid_score > best_valid:
                            best_valid = valid_score
                            best_i = i
                            self.save("BEST")

                        log_str += "Epoch %d, Step %d: Valid %0.2f\n" % (epoch,i, valid_score * 100.) + \
                                "Epoch %d, Best Step %d: Best %0.2f\n" % (epoch,best_i, best_valid * 100.)

                    print(log_str, end='')

                    with open(self.output + "/log.log", 'a') as f:
                        f.write(log_str)
                        f.flush()

        self.save("LAST")
        return best_valid
    
    def get_masks(self,mode,batch):
        key = mode+str(batch)
        if key in self.mask_cache:
            return self.mask_cache[key]
        
        mask0 = self.mask0.repeat([batch,1])
        mask1 = self.mask1.repeat([batch,1])
        mask2 = self.mask2.repeat([batch,1])
        mask3 = self.mask3.repeat([batch,1])
        
        self.mask_cache[key] = [mask0,mask1,mask2,mask3]
        return self.mask_cache[key]

    def calculatelogits(self,anspreds,typepreds,mode):
            batch = anspreds.size()[0]
            mask0,mask1,mask2,mask3 = self.get_masks(mode,batch)
            replen = len(self.train_tuple.dataset.label2ans)
            anspreds0 = anspreds*mask0*typepreds.select(dim=1,index=0).reshape([batch,1]).repeat([1,replen])
            anspreds1 = anspreds*mask1*typepreds.select(dim=1,index=1).reshape([batch,1]).repeat([1,replen])
            anspreds2 = anspreds*mask2*typepreds.select(dim=1,index=2).reshape([batch,1]).repeat([1,replen])
            anspreds3 = anspreds*mask3*typepreds.select(dim=1,index=2).reshape([batch,1]).repeat([1,replen])
            nanspreds=anspreds0+anspreds1+anspreds2+anspreds3
            return nanspreds
    
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
        
        type_accuracy = 0.0
        num_batches=0
        for i, datum_tuple in tqdm(enumerate(loader),ascii=True,desc="Evaluating"):
            ques_id, feats, boxes, sent, typed_target, target = datum_tuple   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit,typelogit = self.model(feats, boxes, sent)
                
                logit = self.sigmoid(logit)
                type_logit_soft = self.softmax(typelogit)
                logit = self.calculatelogits(logit,type_logit_soft,"predict")
                
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
                    
#               type_accuracy+= torch.mean(torch.eq((typelogit>0.5).int().cuda(),typed_target.cuda()).float().cuda()).cpu().item()
                type_accuracy+= torch.mean(torch.eq(typelogit.argmax(dim=1).cuda(),typed_target.cuda()).float().cuda()).cpu().item()
                num_batches+=1
        
        print("Type Accuracy:",type_accuracy/num_batches)
                
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
            ques_id, feats, boxes, sent, typetarget, target = datum_tuple
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
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False,folder=args.data),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test and not args.dump:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            for data in ["orig","vqaorigyn","vqasinglelol","vqamultilol","vqalolval","vqacwlsingle","vqacwlmulti","vqacwlval","varlenlol","varlencwl","comma"]:
                print("Data folder:",data)
                args.data=data
                result = vqa.evaluate(
                    get_data_tuple('minival', bs=950,
                                   shuffle=False, drop_last=False,folder=args.data),
                    dump=os.path.join(args.output, 'minival_predict.json')
                )
                print("Current Result:"+str(result))
                print("\n\n\n\n\n")
        elif 'val' in args.test and args.dump:
            vqa.predict(
                    get_data_tuple("minival", bs=950,
                                   shuffle=False, drop_last=False,folder=args.data),
                    dump=os.path.join('data/vqa/'+args.data, 'typed_val_predict.json')
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
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100),flush=True)
        else:
            print("DO NOT USE VALIDATION")
        best_valid= vqa.train(vqa.train_tuple, vqa.valid_tuple)
        
        #Send Mail when training ends
        hostname = socket.gethostname()
        msg = MIMEText(str(args)+"\n HostName:"+str(hostname)+"\n")
        msg["From"] = "pbanerj6@asu.edu"
        msg["To"] = "pbanerj6@asu.edu"
        msg["Subject"] = "Job has ended. Valid Score:" + str(best_valid)
        p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
        p.communicate(msg.as_bytes())

    
