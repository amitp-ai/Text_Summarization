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
import torch.optim.lr_scheduler as lrSched
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from datetime import datetime
import evaluate
import copy
import argparse

logger = utils.create_logger('train.log')

def train(model, train_data, val_data, abs_idx2word, device, batch_size, 
        num_epochs, print_every_iters, lr, tb_descr, savedModelDir=None, step=0, bestMetricVal=None, l2Reg=0.0):
    '''This is the main function for model training'''

    #data setup
    train_data.move_to(torch.device('cpu')) #keep data on cpu
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0) #set num_workers to #cpu cores?? But 0 seems fastest in my experimentation
    val_data.move_to(torch.device('cpu')) #keep data on cpu
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    #model to cpu/gpu
    model = model.to(device=device)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2Reg)

    #lr scheduler
    def lrLambdaFunc(step):
        mult = 1
        if step > 5000: 
            mult=0.8
        elif step > 10000: 
            mult=0.7
        elif step > 20000:
            mult=0.5
        elif step > 30000:
            mult=0.25
        return mult
    scheduler = lrSched.LambdaLR(optimizer, lrLambdaFunc) #lr decay

    #tensorboard setup
    logdir = f"runs/seq2seqWithAtten/{datetime.now().strftime('%Y%m%d-%H%M%S')}_lr-{lr}_{tb_descr}"
    tb_writer = SummaryWriter(logdir)

    #modify abs_idx2word by removing pad tokens so as to correctly calculate Reouge scores
    abs_idx2word[0] = ''

    #checkpoint saver
    ckptSaver = utils.CheckpointSaver(saveDir=savedModelDir, metricName='rouge-1', maximizeMetric=True, logger=logger, bestVal=bestMetricVal)

    step = 0
    model.train()
    for e in range(num_epochs):
        for x,yt,_,_,_ in train_dataloader:
            x, yt = x.to(device), yt.to(device)
            optimizer.zero_grad()
            y = yt[:,:-1] #yt starts with --start-- token and ends with --stop-- or --pad-- tokens.
            # logger.debug(y.shape, yt.shape)
            y = model(x, y)
            seq_len = y.shape[2]+1
            ygt = yt[:,1:seq_len]
            # logger.debug(y.shape, ygt.shape)
            loss = F.cross_entropy(y, ygt.to(torch.int64))
            loss.backward()
            optimizer.step()
            lrRate = optimizer.state_dict()['param_groups'][0]['lr']
            
            if (step % print_every_iters) == 0:
                logger.debug(f'After Iteration {step}: LR is {lrRate} and Loss is {loss.item():.6f}')
                #evaluate on training data
                logger.debug(f'\tModel eval on training data after iteration {step}...')
                r1, r2, rl = evaluate.evaluate_model(model, train_dataloader, abs_idx2word, device)
                logger.debug(f'\t\tRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')
                tb_writer.add_scalars('Training Data Rouge Scores', {'rouge-1':r1, 'rouge-2':r2, 'rouge-l':rl}, step)

                #evaluate on validation data
                logger.debug(f'\tModel eval on validation data after iteration {step}...')
                r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device)
                logger.debug(f'\t\tRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')
                tb_writer.add_scalars('Validation Data Rouge Scores', {'rouge-1':r1, 'rouge-2':r2, 'rouge-l':rl}, step)

                #save model checkpoint
                ckptSaver.save(step=step, model=model, metricVal=r1, device=device)
                
            #log the running loss
            tb_writer.add_scalar('Training Loss', loss.item(), step)

            # LR scheduler
            scheduler.step() #call lr scheduler every iteration
            step += 1

    #final model evaluation and saving checkpoint
    logger.debug(f'\nModel eval on validation data after final iteration...')
    r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device, print_example=True)
    logger.debug(f'\n\tRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')
    ckptSaver.save(step=step, model=model, metricVal=r1, device=device)
    
    #visualize the model
    with torch.no_grad():
        x = torch.randint(0, 10, size=(3,5)).to(device=device, dtype=torch.int32)
        yt = torch.randint(0, 10, size=(3,5)).to(device=device, dtype=torch.int32)
        tb_writer.add_graph(model, (x,yt), verbose=False) #write to tensorboard
    tb_writer.flush()
    tb_writer.close()


def evaluateModel(model, val_data, abs_idx2word, device, batch_size):
    """ To evaluate the model """
    #modify abs_idx2word by removing pad tokens so as to correctly calculate Reouge scores
    abs_idx2word[0] = ''

    #data setup
    val_data.move_to(torch.device('cpu')) #keep data on cpu
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    #model instantiation
    model = model.to(device=device)
    #evaluation
    logger.debug(f'\tModel eval on validation data...')
    r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device, print_example=True)
    logger.debug(f'\nRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')



def get_data(use_full_vocab=True, cpc_codes='de', fname='data0_str_json.gz', train_size=128, val_size=16):
    """Get data for training"""
    logger.debug('Getting the training and validation data...')
    #for desc and abs: get the vocab, word2idx, idx2word
    desc_vocab = utils.load_vocab(file_name = f"../DataWrangling/desc_vocab_final_{cpc_codes}_after_preprocess_text.json")
    abs_vocab = utils.load_vocab(file_name = f"../DataWrangling/abs_vocab_final_{cpc_codes}_after_preprocess_text.json")

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
    logger.debug(f'Size of description vocab is {len(lang_train.desc_vocab)} and abstract vocab is {len(lang_train.abs_vocab)}')
    lang_val = utils.Mini_Data_Language_Info(mini_df_val, desc_word2idx=lang_train.desc_word2idx,abs_word2idx=lang_train.abs_word2idx)

    train_data = utils.bigPatentDataset(lang_train.mini_data, shuffle=True)
    train_data.memory_size()
    val_data = utils.bigPatentDataset(lang_val.mini_data, shuffle=True)

    return train_data, val_data, lang_train

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_train_args():
    """ Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train a Text Summarization Model')

    # parser.add_argument('--hiddenDim', nargs='?', type=int, default=50, help='The size of the hidden dimension to be used for all layers')
    parser.add_argument('--hiddenDim', type=int, default=50, help='The size of the hidden dimension to be used for all layers')
    parser.add_argument('--numLayers', type=int, default=2, help='The number of LSTM layers')
    parser.add_argument('--batchSize', type=int, default=16, help='The batch size')
    parser.add_argument('--numEpochs', type=int, default=10, help='The number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=0, help='To seed the random state for repeatability')
    parser.add_argument('--printEveryIters', type=int, default=250, help='To print/log after this many iterations')
    parser.add_argument('--tbDescr', type=str, default='', help='Experiment description for tensorboard logging')
    parser.add_argument('--savedModelDir', default=None, help='Location for saving model checkpoints during training')
    parser.add_argument('--loadBestModel', type=str2bool, default=False, help='Load the Best Saved Model')
    parser.add_argument('--modelType', type=str, help='The Model Type to Use')
    parser.add_argument('--toTrain', type=str2bool, default=True, help='Flag to train or evaluate the model')
    parser.add_argument('--fullVocab', type=str2bool, default=True, help='Use the full vocab or generate vocab based only upon the data used for training')
    parser.add_argument('--trainSize', type=int, default=128, help='Size of training data')
    parser.add_argument('--valSize', type=int, default=16, help='Size of validation data')
    parser.add_argument('--tfThresh', type=float, default=0.0, help='Amount of time to not do teacher forcing during training')
    parser.add_argument('--beamSize', type=int, default=0, help='Beam size for beam search. 0 means no beam search')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_train_args()
    logger.debug(args)

    #random state setup (for repeatability)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed) #no need to do this
    # torch.backends.cudnn.deterministic = True #not sure if need this
    # torch.backends.cudnn.benchmark = False #not sure if need this

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, lang_train = get_data(use_full_vocab=args.fullVocab, cpc_codes='de', fname='data0_str_json.gz', 
                                                train_size=args.trainSize, val_size=args.valSize)
    model = eval(args.modelType)(descVocabSize=len(lang_train.desc_vocab), absVocabSize=len(lang_train.abs_vocab), 
                                hiddenDim=args.hiddenDim, numLayers=args.numLayers, dropout=args.dropout, 
                                tfThresh=args.tfThresh, beamSize=args.beamSize)
    step = 0
    metricVal = -1
                
    #load best saved model
    if args.loadBestModel:
        model, step, metricVal = utils.loadModel(model, f'{args.savedModelDir}/best.pth.tar', device, return_step=True)
        logger.debug(f'Loaded the current best model for {model.__class__.__name__}, which is from step {step} and metric value is {metricVal:.3f}')

    #evaluate or train the model
    if not args.toTrain:
        logger.debug('Starting model evaluation for the current best model...')
        evaluateModel(model=model, val_data=val_data, abs_idx2word=lang_train.abs_idx2word, 
                device=device, batch_size=args.batchSize)
        # utils.profileModel(model, val_data, devName='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        logger.debug('\nStarting model training...')
        train(model=model, train_data=train_data, val_data=val_data, abs_idx2word=lang_train.abs_idx2word, 
        device=device, batch_size=args.batchSize, num_epochs=args.numEpochs, lr=args.lr, 
        print_every_iters=args.printEveryIters, tb_descr=args.tbDescr, savedModelDir=args.savedModelDir, 
        step=step, bestMetricVal=metricVal)
