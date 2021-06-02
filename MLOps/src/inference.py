"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Main code for inference
"""

import torch
import argparse
import utils
import models
import loadAndPreprocessData
import pandas as pd
import os
from rouge import Rouge
import time

PARENT_DIR = './'

def modelInference(model, descData, abs_idx2word, device, logger):
    """ To evaluate the model on a single input at a time ***(i.e. not batched input)
    descData is of shape: 1xSeqLen ***"""

    #data setup
    # descData.move_to(torch.device('cpu')) #keep data on cpu

    model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        x,ytgt = descData[0:1]
        logger['Description_Length'] = x.shape[1]-1 #remove stop token
        x = x.to(device).to(torch.int32)
        y = model.evaluate(x) #y is an int32 tensor of shape 1xL 
        logger['Prediction_Length'] = y.shape[1]-2 #remove start and stop tokens
        pred = pd.Series(y[0,:].cpu()).map(abs_idx2word).tolist()
        pred = ' '.join(pred[1:-1]) #remove start and stop tokens
        logger['Prediction_Summary'] = pred
        rouge_score = None
        if ytgt != None:
            rouge_evaluator = Rouge()
            target = logger['TgtSmry_AfterPreProcess']
            target = ' '.join(target)
            rouge_score = rouge_evaluator.get_scores(pred, target) #pred is the first argument (https://pypi.org/project/rouge/)
        logger['Rouge_Scores'] = rouge_score
        print(pred, rouge_score)
    # model.train()
    return logger


def get_args():
    """ Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Inference for Text Summarization Model')

    parser.add_argument('--hiddenDim', type=int, default=50, help='The size of the hidden dimension to be used for all layers')
    parser.add_argument('--numLayers', type=int, default=2, help='The number of Enc layers')
    parser.add_argument('--decNumLayers', type=int, default=4, help='The number of Dec layers')
    parser.add_argument('--numHeads', type=int, default=4, help='The number of Mutihead Attention Heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--loadModelName', help='Load a Saved Model')
    parser.add_argument('--inputTextFile', help='Load the input data stored in json file')
    parser.add_argument('--modelType', type=str, help='The Model Type to Use')
    parser.add_argument('--beamSize', type=int, default=0, help='Beam size')
    parser.add_argument('--configPath', type=str, default='', help='File path to config parameters')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    t0 = time.time()
    args = get_args()
    logger = utils.CSVLogger()
    logger['Time_Stamp'] = time.strftime("%H:%M:%S on %Y/%m/%d")
    logger['args'] = args
    config = utils.read_params(args.configPath)['Inference']
    cfgParams = config['OtherParams']
    cfgModel = config['Models'][args.modelType]

    #random state setup (for repeatability)
    torch.backends.cudnn.deterministic = True #not sure if need this (#ignored if GPU not available)
    torch.manual_seed(cfgParams['seed'])
    torch.cuda.manual_seed_all(cfgParams['seed']) #no need to do this (#ignored if GPU not available)
    # torch.backends.cudnn.benchmark = False #not sure if need this

    #this is the inference pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1 = time.time()
    descData, descVocabSize, absVocabSize, absIdx2Word, logger = loadAndPreprocessData.getData(inputTextFile=args.inputTextFile, 
                                                        cpc_codes=cfgParams['cpcCodes'], logger=logger)
    t2 = time.time()
    model = eval('models.'+args.modelType)(descVocabSize=descVocabSize, absVocabSize=absVocabSize, 
                                beamSize=args.beamSize, embMult=cfgModel['embMult'], predMaxLen=cfgModel['predMaxLen'], 
                                encMaxLen=cfgModel['encMaxLen'], pad_token=cfgParams['padToken'], 
                                hiddenDim=args.hiddenDim, numLayers=args.numLayers, dropout=args.dropout,
                                numHeads=args.numHeads, decNumLayers=args.decNumLayers)
    #load model
    model, step, metricVal = utils.loadModel(model, f'{args.loadModelName}', device, return_step=True)
    logger['Model_Info'] = f'Loaded {args.loadModelName} model for {model.__class__.__name__}, which is from step {step} and metric value is {metricVal:.3f}'
    t3 = time.time()

    #evaluate
    print('Running Inference...')
    logger = modelInference(model=model, descData=descData, abs_idx2word=absIdx2Word, device=device, logger=logger)
    t4 = time.time()
    # utils.profileModel(model, val_data, devName='cuda' if torch.cuda.is_available() else 'cpu')

    #log duration
    logger['Data Loading and Preprocessing Duration (s)'] = round(t2-t1, 3)
    logger['Model Loading Duration (s)'] = round(t3-t2, 3)
    logger['Model Inference Duration (s)'] = round(t4-t3, 3)
    logger['Total Inference Duration (s)'] = round(t4-t0, 3)
    logger.toCSV(PARENT_DIR+'Data/inference.csv')


