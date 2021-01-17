import json
import os, sys
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


ANS2LABEL_PATH = "data/mutant_both_vqacp_v2/mutant_cp_merge_ans2label.json"
LABEL2ANS_PATH = "data/mutant_both_vqacp_v2/mutant_cp_merge_label2ans.json"
INDEXLIST = json.load(open("./data/mutant_both_vqacp_v2/mutant_merge_indexlist.json"))
ans2label = json.load(open(ANS2LABEL_PATH))
label2ans = json.load(open(LABEL2ANS_PATH))

indextensor = torch.cuda.LongTensor(INDEXLIST)
MASK0 = torch.eq(indextensor,0).float()
MASK1 = torch.eq(indextensor,1).float()
MASK2 = torch.eq(indextensor,2).float()
MASK3 = torch.eq(indextensor,3).float()

mask_cache = {}



def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def get_masks(mode,batch):
    key = mode+str(batch)
    if key in mask_cache:
        return mask_cache[key]
    
    # print(batch)
    mask0 = Variable(MASK0.repeat(batch,1)).cuda()
    mask1 = Variable(MASK1.repeat(batch,1)).cuda()
    mask2 = Variable(MASK2.repeat(batch,1)).cuda()
    mask3 = Variable(MASK3.repeat(batch,1)).cuda()
    
    mask_cache[key] = [mask0,mask1,mask2,mask3]
    return mask_cache[key]

def calculatelogits(anspreds, typepreds, mode):
        batch = anspreds.size()[0]
        # print(anspreds.size(), batch, type(batch))
        mask0,mask1,mask2,mask3 = get_masks(mode,batch)
        replen = len(label2ans)
        # print("type(anspreds), type(mask0), type(typepreds)", 
        #       type(anspreds), type(mask0), type(typepreds))

        anspreds, typepreds = anspreds.data , typepreds.data 
        mask0, mask1, mask2, mask3 = mask0.data, mask1.data, mask2.data, mask3.data

        anspreds0 = anspreds*mask0*typepreds.select(1,0).unsqueeze(1).repeat(1,replen)
        anspreds1 = anspreds*mask1*typepreds.select(1,1).unsqueeze(1).repeat(1,replen)
        anspreds2 = anspreds*mask2*typepreds.select(1,2).unsqueeze(1).repeat(1,replen)
        anspreds3 = anspreds*mask3*typepreds.select(1,3).unsqueeze(1).repeat(1,replen)

        nanspreds = anspreds0 + anspreds1 + anspreds2 + anspreds3
        return nanspreds


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch, tiny_train=False):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []

    total_step = 0

    ans_embed = np.load("./data/mutant_both_vqacp_v2/answer_embs.npy") +1e-8
    ans_embed = torch.from_numpy(ans_embed).cuda()
    ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()

        model.train(True)


        cos = nn.CosineSimilarity()
        type_loss = nn.NLLLoss()

        for i, (v, q, typetarget, a, b, answertypefeats, top_ans_emb) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            if tiny_train and i == 10:
                break
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            typetarget = Variable(typetarget).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            answertypefeats = Variable(answertypefeats).cuda()
            top_ans_emb = Variable(top_ans_emb).cuda()

            gen_embs, pred, type_logit, loss, all_ans_embs = model(v, None, q, a, b)
            all_ans_embs  = torch.stack([all_ans_embs]*gen_embs.shape[0])            

            if (loss != loss).any():
              raise ValueError("NaN loss")

            # ## NCE LOSS            
            # positive_dist = cos(gen_embs, top_ans_emb) # shape b,k;b,k-> b
            # gen_embs = torch.cat([gen_embs.unsqueeze(1)]*all_ans_embs.shape[1],dim=1)
            # d_logit = cos(gen_embs,all_ans_embs)
            # num = torch.exp(positive_dist).squeeze(-1)
            # den = torch.exp(d_logit).sum(-1)
            # loss_nce = -1 *torch.log(num/den)
            # loss_nce  = loss_nce.mean() * d_logit.size(1)

            ## TYPE LOSS
            logit = nn.functional.sigmoid(pred)
            type_logit_soft = nn.functional.softmax(type_logit)
            type_logit = nn.functional.log_softmax(type_logit)
            
            logit = calculatelogits(logit, answertypefeats, "train")
            loss_type = type_loss(type_logit, typetarget)
            loss_type = loss_type


            loss = loss + loss_type
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        if tiny_train:
            L = total_step 
        else:
            L = len(train_loader.dataset)

        total_loss /= L
        train_score = 100 * train_score / L

        run_eval = eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, tiny_train)
            results["epoch"] = epoch+1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            eval_score = results["score"]
            bound = results["upper_bound"]
            print("EVAL SCORE", eval_score)
            print("UPPER BOUNG", bound)
            # print("TYPEWISE SCORE", results["type_acc"])
            all_results.append(results)

            with open(join(output, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            model.train(True)

            

        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))


    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader, tiny_eval=False):
    score = 0
    type_score = torch.zeros(4, 1).cuda()
    type_count = torch.zeros(4, 1).cuda()
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    count = 0
    # score_bytype = {"yes/no": 0, "number": 0, "other": 0}
    # count_bytype = {"yes/no": 0, "number": 0, "other": 0}
    # acc_bytype = {"yes/no": 0, "number": 0, "other": 0}

    for v, q, _, a, _, afeats, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        afeats = afeats.cuda()
        afeats = torch.t(afeats) # 4 x B

        gen_embs, pred, _, _, _ = model(v, None, q, None, None)
        all_logits.append(pred.data.cpu().numpy())

        batch_score = compute_score_with_logits(pred, a.cuda()).sum(1).unsqueeze(1) # B x 1

        # print(afeats.size(), batch_score.size())

        type_score += torch.mm(afeats, batch_score)
        type_count += torch.sum(afeats, 1).unsqueeze(1)
    
        batch_score_sum = batch_score.sum()
        score += batch_score_sum
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

        count += 1

        if tiny_eval:
            if count == 10:
                break

    if tiny_eval:
        L = count*512
    else:
        L = len(dataloader.dataset)

    score = score / L
    upper_bound = upper_bound / L
    type_acc = torch.div(type_score, type_count + 1)
    print("upper_bound", upper_bound, "eval score", score, "type_acc", type_acc)
    sys.stdout.flush()



    results = dict(
        score=score,
        upper_bound=upper_bound, 
        acc0=type_acc.tolist()[0],
        acc1=type_acc.tolist()[1],
        acc2=type_acc.tolist()[2],
        acc3=type_acc.tolist()[3], 
        num0=type_count.tolist()[0],
        num1=type_count.tolist()[1],
        num2=type_count.tolist()[2],
        num3=type_count.tolist()[3]
    )
    return results
