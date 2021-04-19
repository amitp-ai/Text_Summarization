"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    ***DON'T USE THIS. USE train.py INSTEAD***
    This is used only for model experimentation submission for step7.
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
import argparse


def train(model, train_data, val_data, abs_idx2word, device, batch_size, 
        num_epochs, print_every_iters, lr, tb_descr, seed=0):
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



def get_data(use_full_vocab=True, cpc_codes='de', fname='data0_str_json.gz', train_size=128, val_size=16):
    print('Getting the training data...')
    #for desc and abs: get the vocab, word2idx, idx2word
    desc_vocab = utils.load_vocab(file_name = f"../DataWrangling/desc_vocab_final_{cpc_codes}_after_preprocess_text.json")
    abs_vocab = utils.load_vocab(file_name = f"../DataWrangling/abs_vocab_final_{cpc_codes}_after_preprocess_text.json")
    print(f'Size of description vocab is {len(desc_vocab)} and abstract vocab is {len(abs_vocab)}')

    desc_word2idx = utils.load_word2idx(file_name = f'../DataWrangling/desc_{cpc_codes}_word2idx.json')
    abs_word2idx = utils.load_word2idx(file_name = f'../DataWrangling/abs_{cpc_codes}_word2idx.json')

    desc_idx2word = utils.load_idx2word(file_name = f'../DataWrangling/desc_{cpc_codes}_idx2word.json')
    abs_idx2word = utils.load_idx2word(file_name = f'../DataWrangling/abs_{cpc_codes}_idx2word.json')

    #get the training and val data
    data_train = utils.load_data_string(split_type='train', cpc_codes=cpc_codes, fname=fname)
    data_val = utils.load_data_string(split_type='val', cpc_codes=cpc_codes, fname=fname)
    mini_df_train = utils.get_mini_df(data_train, mini_df_size=train_size) 
    mini_df_val = utils.get_mini_df(data_val, mini_df_size=val_size) 

    if use_full_vocab:
        lang_train = utils.Mini_Data_Language_Info(mini_df_train, desc_word2idx=desc_word2idx,abs_word2idx=abs_word2idx,
                                                desc_idx2word=desc_idx2word, abs_idx2word=abs_idx2word,
                                                desc_vocab=desc_vocab, abs_vocab=abs_vocab) #using full vocab
    else:
        lang_train = utils.Mini_Data_Language_Info(mini_df_train) #generate vocab etc (i.e. don't use full vocab)
    lang_val = utils.Mini_Data_Language_Info(mini_df_val, desc_word2idx=lang_train.desc_word2idx,abs_word2idx=lang_train.abs_word2idx)

    train_data = utils.bigPatentDataset(lang_train.mini_data, shuffle=True)
    train_data.memory_size()
    val_data = utils.bigPatentDataset(lang_val.mini_data, shuffle=True)

    return train_data, val_data, lang_train


def get_train_args():
    """ Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train a Text Summarization Model')

    parser.add_argument('--hidden_dim', nargs='?', type=int, default=50, help='The size of the hidden dimension to be used for all layers')
    parser.add_argument('--num_layers', nargs='?', type=int, default=2, help='The number of LSTM layers')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, help='The batch size')
    parser.add_argument('--num_epochs', nargs='?', type=int, default=10, help='The number of epochs to train for')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--seed', nargs='?', type=int, default=0, help='To seed the random state for repeatability')
    parser.add_argument('--print_every_iters', nargs='?', type=int, default=250, help='To print/log after this many iterations')
    parser.add_argument('--tb_descr', nargs='?', type=str, default='', help='Experiment description for tensorboard logging')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, lang_train = get_data(use_full_vocab=True, cpc_codes='de', fname='data0_str_json.gz', 
                                                train_size=128, val_size=16)
    encoder = models.EncoderLSTM(vocab_size=len(lang_train.desc_vocab), hidden_dim=args.hidden_dim, 
                                num_layers=args.num_layers, bidir=True)
    decoder = models.DecoderLSTM(vocab_size=len(lang_train.abs_vocab), hidden_dim=args.hidden_dim, 
                                num_layers=args.num_layers, bidir=False)
    model = models.Seq2Seq(encoder, decoder)

    #delete this
    train_data.shuffle(2)
    val_data.shuffle(2)

    print('\nStarting model training...')
    print(args)
    train(model=model, train_data=train_data, val_data=val_data, 
                abs_idx2word=lang_train.abs_idx2word, device=device, batch_size=args.batch_size, 
                num_epochs=args.num_epochs, lr=args.lr, print_every_iters=args.print_every_iters, 
                tb_descr=args.tb_descr, seed=args.seed)
