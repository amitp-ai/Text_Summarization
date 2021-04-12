"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Main code to train the text summarizer
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from datetime import datetime
import evaluate
import copy

def train(model, train_data, val_data, abs_idx2word, device, batch_size=4, num_epochs=10, 
        print_every_iters=25, lr=1e-3, seed=101, tb_descr=''):
    '''
    This is the main function for model training
    '''

    #random state setup (for repeatability)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) #no need to do this
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #data setup
    train_data.move_to(torch.device('cpu')) #keep data on cpu
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0) #set num_workers to #cpu cores?? But 0 seems fastest in my experimentation
    val_data.move_to(torch.device('cpu')) #keep data on cpu
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    #model instantiation
    model = model.to(device=device)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #tensorboard setup
    logdir = f"runs/enc_dec_lstm/{datetime.now().strftime('%Y%m%d-%H%M%S')}_lr-{lr}_{tb_descr}"
    tb_writer = SummaryWriter(logdir)

    #modify abs_idx2word by removing pad tokens so as to correctly calculate Reouge scores
    abs_idx2word = copy.deepcopy(abs_idx2word)
    abs_idx2word[0] = ''

    iter_ = 0
    for e in range(num_epochs):
        for x,yt,_,_ in train_dataloader:
            x, yt = x.to(device), yt.to(device)
            optimizer.zero_grad()
            y = yt[:,:-1]
            # print(y.shape, yt.shape)
            y = model(x, y)
            seq_len = y.shape[2]+1
            ygt = yt[:,1:seq_len]
            # print(y.shape, ygt.shape)
            loss = F.cross_entropy(y, ygt.to(torch.int64))
            loss.backward()
            optimizer.step()

            if (iter_) % print_every_iters == 0:
                print(f'After Iteration {iter_}, Loss is: {loss.item():.6f}')

                #evaluate model
                print(f'\tModel eval on training data after iteration {iter_}...')
                r1, r2, rl = evaluate.evaluate_model(model, train_dataloader, abs_idx2word, device)
                print(f'\t\tRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')
                tb_writer.add_scalars('Training Data Rouge Scores', {'rouge-1':r1, 'rouge-2':r2, 'rouge-l':rl}, iter_)

                print(f'\tModel eval on validation data after iteration {iter_}...')
                r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device)
                print(f'\t\tRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')
                tb_writer.add_scalars('Validation Data Rouge Scores', {'rouge-1':r1, 'rouge-2':r2, 'rouge-l':rl}, iter_)

            #log the running loss
            tb_writer.add_scalar('Training Loss', loss.item(), iter_)
            iter_ += 1

    #final model evaluation
    print(f'\nModel eval on validation data after final iteration...')
    evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device, print_example=True)
    
    #visualize the model
    with torch.no_grad():
        x = torch.randint(0, 10, size=(3,5)).to(device=device, dtype=torch.int32)
        yt = torch.randint(0, 10, size=(3,5)).to(device=device, dtype=torch.int32)
        tb_writer.add_graph(model, (x,yt), verbose=False) #write to tensorboard
    tb_writer.flush()
    tb_writer.close()

