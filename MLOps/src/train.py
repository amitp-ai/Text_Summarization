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
from datetime import datetime
import copy
import argparse
import utils
import models
import evaluate
from tqdm import tqdm
import wandb

wandb.login()
logger = utils.create_logger('train.log')

def train(model, train_data, val_data, abs_idx2word, device, batch_size, num_epochs, 
    print_every_iters, lr, savedModelBaseName, l2Reg, modelArtifact, wandbRun, step=0, bestMetricVal=None):
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
        if step < 2500: 
            mult=1.0
        elif step < 5000: 
            mult=0.75
        elif step < 7500:
            mult=0.5
        elif step < 10000:
            mult=0.25
        else:
            mult=0.1
        return mult
    LR_scheduler = lrSched.LambdaLR(optimizer, lrLambdaFunc) #lr decay

    #modify abs_idx2word by removing pad tokens so as to correctly calculate Reouge scores
    abs_idx2word[0] = ''

    #checkpoint saver
    ckptSaver = utils.CheckpointSaver(savedModelBaseName=savedModelBaseName, metricName='rouge-1', 
                        maximizeMetric=True, logger=logger, bestVal=bestMetricVal)

    #create wandb text table to store predicted summary
    predSmryArtifact = wandb.Artifact(name='tgtAndPredExample', type='trainAndValSamples')
    wandbTrainTextTable = wandb.Table(columns=["Example Target", "Example Prediction", "Example Rouge Scores"])
    wandbValTextTable = wandb.Table(columns=["Example Target", "Example Prediction", "Example Rouge Scores"])

    #setup the criterion
    criterion = F.cross_entropy
    
    #watch the parameters and gradients of the model
    wandbRun.watch(model, log='all', log_freq=100)

    step = 0
    model.train()
    for e in tqdm(range(num_epochs)):
        for x,yt,_,_,_ in train_dataloader:
            x, yt = x.to(device), yt.to(device)
            y = yt[:,:-1] #yt starts with --start-- token and ends with --stop-- or --pad-- tokens.
            # logger.debug(y.shape, yt.shape)

            #forward pass
            y = model(x, y)
            seq_len = y.shape[2]+1
            ygt = yt[:,1:seq_len]
            # logger.debug(y.shape, ygt.shape)
            loss = criterion(y, ygt.to(torch.int64))

            #backward pass
            optimizer.zero_grad()
            loss.backward()

            #update weights
            optimizer.step()

            lrRate = optimizer.state_dict()['param_groups'][0]['lr']
            
            if (step % print_every_iters) == 0:
                wandbRun.log({'Learning Rate': lrRate, 'Loss': loss.item()}, step=step)
                #evaluate on training data
                r1, r2, rl = evaluate.evaluate_model(model, train_dataloader, abs_idx2word, device, wandbTextTable=wandbTrainTextTable)
                wandbRun.log({'Train_Rouge-1': r1, 'Train_Rouge-2': r2,'Train_Rouge-l': rl}, step=step)

                #evaluate on validation data
                r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device, wandbTextTable=wandbValTextTable)
                wandbRun.log({'Val_Rouge-1': r1, 'Val_Rouge-2': r2,'Val_Rouge-l': rl}, step=step)

                #save model checkpoint
                ckptSaver.save(step=step, model=model, metricVal=r1, device=device)
                
            #log the running loss
            # tb_writer.add_scalar('Training Loss', loss.item(), step)

            # LR scheduler
            LR_scheduler.step() #call lr scheduler every iteration
            step += 1

    #final model evaluation and saving checkpoint
    r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device, wandbTextTable=wandbValTextTable)
    wandbRun.summary['Final_Val_Rouge-1'] = r1
    wandbRun.summary['Final_Val_Rouge-2'] = r2
    wandbRun.summary['Final_Val_Rouge-l'] = rl
    ckptSaver.save(step=step, model=model, metricVal=r1, device=device)

    #add the saved models to wandb
    ckptSaver.addChkptsToArtifact(modelArtifact, wandbRun)

    #add and log text table to wandb artifact
    predSmryArtifact.add(wandbTrainTextTable, name='trainSamples')
    predSmryArtifact.add(wandbValTextTable, name='validationSamples')
    wandbRun.log_artifact(predSmryArtifact)


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
    r1, r2, rl = evaluate.evaluate_model(model, val_dataloader, abs_idx2word, device)
    logger.debug(f'\nRouge-1 is {r1:.4f}, Rouge-2 is {r2:.4f}, and Rouge-l is {rl:.4f}')


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
    parser.add_argument('--hiddenDim', type=int, default=128, help='The size of the hidden dimension to be used for all layers')
    parser.add_argument('--numLayers', type=int, default=2, help='The number of Enc layers')
    parser.add_argument('--decNumLayers', type=int, default=4, help='The number of Dec layers')
    parser.add_argument('--numHeads', type=int, default=4, help='The number of Mutihead Attention Heads')
    parser.add_argument('--batchSize', type=int, default=16, help='The batch size')
    parser.add_argument('--numEpochs', type=int, default=10, help='The number of epochs to train for')
    parser.add_argument('--beamSize', type=int, default=0, help='Beam size')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--loadBestModel', type=str2bool, default=False, help='Load the Best Saved Model')
    parser.add_argument('--modelType', type=str, help='The Model Type to Use')
    parser.add_argument('--savedModelBaseName', default='MODEL7', help='Base name for saving model checkpoints during training')
    parser.add_argument('--configPath', type=str, default='', help='File path to config parameters')

    args = parser.parse_args()
    return args


def Pipeline(cfg):
    #Setup WandB
    wandbRun = wandb.init(project="Text-Summarization", notes="model training", tags=["train", "learning"], 
    config=cfg, save_code=False, job_type='MODEL1', group='Train')
    config = wandbRun.config

    #Random State Setup (for repeatability)
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True #not sure if need this (#ignored if GPU not available)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed']) #no need to do this (#ignored if GPU not available)
    # torch.backends.cudnn.benchmark = False #not sure if need this
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load Data
    if not config.loadFromArtifact:
        train_data, val_data, lang_train = utils.get_data(use_full_vocab=config['fullVocab'], 
                                cpc_codes=config['cpcCodes'], fname=config['fname'], 
                                train_size=config['trainSize'], val_size=config['valSize'], logger=logger)
    else:
        logger.debug('Loading data from wandb artifact...')
        train_data, val_data, lang_train = utils.loadWandBDataArtifact(artifactName='Data', wandbRun=wandbRun, version='latest')
    
    if not config.loadFromArtifact:
        #Artifact to save data to wandb for repeatability
        dataConfig = ['trainSize', 'valSize', 'fullVocab', 'cpcCodes', 'fname']
        dataConfig = {key:config[key] for key in dataConfig}
        dataArtifact = wandb.Artifact(name='Data', type='data',
            description='Training data, val data, and data dicts used', metadata=dataConfig)
        utils.saveWandBDataArtifact(train_data=train_data, val_data=val_data, lang_train=lang_train, 
                                    dataArtifact=dataArtifact, wandbRun=wandbRun)

    #Build and Load model
    model = eval('models.'+config.modelType)(descVocabSize=len(lang_train.desc_vocab), absVocabSize=len(lang_train.abs_vocab), 
                                beamSize=config.beamSize, embMult=config['embMult'], predMaxLen=config['predMaxLen'], 
                                encMaxLen=config['encMaxLen'], pad_token=config['padToken'], 
                                hiddenDim=config.hiddenDim, numLayers=config.numLayers, dropout=config.dropout,
                                numHeads=config.numHeads, decNumLayers=config.decNumLayers)
    step = 0
    metricVal = -1
    if config.loadBestModel: #load best saved model
        logger.debug('Loading model from wandb artifact...')
        # model, step, metricVal = utils.loadModel(model, f'{config.savedModelBaseName}_best.pt', device, return_step=True)
        model, step, metricVal = utils.loadWandBModelArtifact(artifactNameVer=f"{config.savedModelBaseName}:latest", 
            wandbRun=wandbRun, model=model, device=device, return_step=True)
        logger.debug(f'Loaded the current best model for {model.__class__.__name__}, which is from step {step} and metric value is {metricVal:.3f}')
    #Artifact to save models to wandb for repeatability
    modelConfig = cfg
    modelArtifact = wandb.Artifact(name=config.savedModelBaseName, type='model',
        description=config.modelType, metadata=modelConfig)

    #Train or Evaluate the Model
    if config['toTrain']:
        logger.debug('\nStarting model training...')
        train(model=model, train_data=train_data, val_data=val_data, abs_idx2word=lang_train.abs_idx2word, 
            device=device, batch_size=config.batchSize, num_epochs=config.numEpochs, lr=config.lr, 
            print_every_iters=config['printEveryIters'], savedModelBaseName=config.savedModelBaseName, 
            step=step, bestMetricVal=metricVal, l2Reg=config['l2Reg'], wandbRun=wandbRun, modelArtifact=modelArtifact)
    else:
        logger.debug('Starting model evaluation for the current best model...')
        evaluateModel(model=model, val_data=val_data, abs_idx2word=lang_train.abs_idx2word, 
                device=device, batch_size=config.batchSize)
        # utils.profileModel(model, val_data, devName='cuda' if torch.cuda.is_available() else 'cpu')
    
    #end wandb
    wandbRun.finish()


if __name__ == '__main__':
    args = get_train_args()
    config = utils.read_params(args.configPath)['Train']
    cfgParams = config['OtherParams']
    cfgModel = config['Models'][args.modelType]
    cfg = {}
    cfg.update(cfgParams)
    cfg.update(cfgModel)
    cfg.update(vars(args))
    logger.debug(cfg)

    Pipeline(cfg)