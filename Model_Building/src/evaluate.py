"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Main code to train the text summarizer
"""

import pandas as pd
import numpy as np
import torch
from rouge import Rouge


def evaluate_helper(y, ygt, yorig, abs_idx2word, print_example):
    '''
    y is an int32 tensor of shape BxVxL (where V is the size of abstract vocabulary)
    ygt is an int32 tensor of shape BxL
    '''
    y = y.cpu().numpy()
    ygt = ygt.cpu().numpy()
    assert y.shape[0] == ygt.shape[0], f'y {y.shape} and ygt {ygt.shape} have different batch sizes'
    batch_size = y.shape[0]
    rouge_1, rouge_2, rouge_l = [], [], []
    for b in range(batch_size):
        pred = pd.Series(y[b,:]).map(abs_idx2word).tolist()
        target = pd.Series(ygt[b,:]).map(abs_idx2word).tolist()
        pred = ' '.join(pred)
        target = ' '.join(target)
        # orig_abs = yorig[b]
        # target = orig_abs
        rouge_evaluator = Rouge()
        rouge_score = rouge_evaluator.get_scores(pred, target) #pred is the first argument (https://pypi.org/project/rouge/)
        rouge_1.append(rouge_score[0]['rouge-1']['f'])
        rouge_2.append(rouge_score[0]['rouge-2']['f'])
        rouge_l.append(rouge_score[0]['rouge-l']['f'])
    rouge_1 = sum(rouge_1)/len(rouge_1)
    rouge_2 = sum(rouge_2)/len(rouge_2)
    rouge_l = sum(rouge_l)/len(rouge_l)
    if print_example: res = (rouge_1, rouge_2, rouge_l, pred, target)
    else: res = (rouge_1, rouge_2, rouge_l)
    return res

def evaluate_model(model, dataloader, abs_idx2word, device, print_example=False):
    rouge_scores = []
    num_samples = 0
    max_num_samples = 16
    with torch.no_grad():
        for x,yt,_,yorig in dataloader:
            num_samples += x.shape[0]
            x, yt = x.to(device), yt.to(device)
            y = yt[:,:-1]
            # y = model(x, y) #teacher forcing
            # y = torch.argmax(y, axis=1).to(dtype=torch.int32) #teacher forcing
            y = model.evaluate(x) #not teacher forcing
            seq_len = y.shape[1]+1
            ygt = yt[:,1:seq_len]
            rouge_scores.append(evaluate_helper(y, ygt, yorig, abs_idx2word, print_example))
            if num_samples > max_num_samples: break
    rouge_1 = sum([r[0] for r in rouge_scores])/len(rouge_scores)
    rouge_2 = sum([r[1] for r in rouge_scores])/len(rouge_scores)
    rouge_l = sum([r[2] for r in rouge_scores])/len(rouge_scores)
    if print_example:
        print(f'Example\nPrediction\n{rouge_scores[0][3]}\nTarget\n{rouge_scores[0][4]}')
        print(f'Rouge-1 is {rouge_scores[0][0]:.4f}, Rouge-2 is {rouge_scores[0][1]:.4f}, and Rouge-l is {rouge_scores[0][2]:.4f}')
    return rouge_1, rouge_2, rouge_l


