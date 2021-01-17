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


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch, tiny_train=False):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []

    total_step = 0

    ans_embed = np.load("data/mutant_only_vqacp_v2/answer_embs.npy") +1e-8
    ans_embed = torch.from_numpy(ans_embed).cuda()
    ans_embed = torch.nn.functional.normalize(ans_embed,dim=1)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()

        model.train(False)
        results = evaluate(model, eval_loader, True)
        eval_score = results["score"]
        bound = results["upper_bound"]
        print("RANDOM SCORE BEFORE TRAINING", eval_score)
        print("UPPER BOUNG", bound)
        model.train(True)


        cos = nn.CosineSimilarity()

        for i, (v, q, a, b, atype, top_ans_emb) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            if tiny_train and total_step == 10:
                break
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()

            _, pred, loss, all_ans_embs = model(v, None, q, a, b)

            # all_ans_embs  = torch.stack([all_ans_embs]*gen_embs.shape[0])

            # print("gen_embs.shape BEFORE unsqueeze", gen_embs.shape)
            # print("all_ans_embs.shape", all_ans_embs.shape)
            # print(type(gen_embs), type(all_ans_embs))
            # sys.stdout.flush()
            

            if (loss != loss).any():
              raise ValueError("NaN loss")

            # ## NCE LOSS
            
            # positive_dist = cos(gen_embs, top_ans_emb) # shape b,k;b,k-> b
            # gen_embs = torch.cat([gen_embs.unsqueeze(1)]*all_ans_embs.shape[1],dim=1)
            # d_logit = cos(gen_embs,all_ans_embs)
            # print("gen_embs.shape AFTER unsqueeze", gen_embs.shape)
            # print("all_ans_embs.shape", all_ans_embs.shape)



            # num = torch.exp(positive_dist).squeeze(-1)
            # den = torch.exp(d_logit).sum(-1)
            # print("num.shape,den.shape", num.shape,den.shape)
            # sys.stdout.flush()
            # loss_nce = -1 *torch.log(num/den)
            # loss_nce  = loss_nce.mean() * d_logit.size(1)

            # loss = loss + loss_nce
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
            results = evaluate(model, eval_loader)
            results["epoch"] = epoch+1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            all_results.append(results)

            with open(join(output, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            eval_bytype_score = results["score_bytype"]

        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            for tt in eval_bytype_score:
                logger.write(tt, eval_bytype_score[tt])

    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader, tiny_eval=False):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    count = 0
    score_bytype = {"yes/no": 0, "number": 0, "other": 0}
    count_bytype = {"yes/no": 0, "number": 0, "other": 0}
    acc_bytype = {"yes/no": 0, "number": 0, "other": 0}

    for v, q, a, b, atype, top_ans_emb in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        _, pred, _, _ = model(v, None, q, None, None)
        all_logits.append(pred.data.cpu().numpy())

        batch_score = compute_score_with_logits(pred, a.cuda())
        for at, bs in zip(atype, batch_score):
            score_bytype[at] += bs 
            count_bytype[at] += 1
            # print(score_bytype)
            # print(count_bytype)

        # print("batch_score", batch_score)


        batch_score_sum = batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        all_bias.append(b)

        count += 1

        if tiny_eval:
            if count == 10:
                break

    if tiny_eval:
        L = count
    else:
        L = len(dataloader.dataset)

    score = score / L

    for tt in ["yes/no", "other", "number"]:
        acc_bytype[tt] = score_bytype[tt]/count_bytype[tt] 
        # print(tt, acc_bytype[tt])

    upper_bound = upper_bound / L


    print("upper_bound", upper_bound, "score", score)

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_bytype=acc_bytype
    )
    return results
