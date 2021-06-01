"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Main code for inference
"""

import torch
import torch.utils.data as data
import argparse
import utils
import models
import evaluate
import json
import pandas as pd
import loadAndPreprocessData

PARENT_DIR = './'
logger = utils.create_logger('inference.log', stdOut=True)


def modelInference(model, descData, abs_idx2word, device):
    """ To evaluate the model on a single input at a time ***(i.e. not batched input)
    descData is of shape: 1xSeqLen ***"""

    #data setup
    # descData.move_to(torch.device('cpu')) #keep data on cpu

    model = model.to(device=device)
    logger.debug(f'\tModel inference...')

    model.eval()
    with torch.no_grad():
        x,ytgt = descData[0:1]
        logger.debug(x.shape)
        x = x.to(device).to(torch.int64)
        y = model.evaluate(x) #y is an int32 tensor of shape 1xVxL (where V is the size of abstract vocabulary) 
        logger.debug(y.shape)
        pred = pd.Series(y[0,:]).map(abs_idx2word).tolist()
        pred = ' '.join(pred)
    # model.train()
    logger.debug(f'Prediction is\n{pred}')


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
    args = get_args()
    logger.debug(args)
    config = utils.read_params(args.configPath)['Inference']
    cfgParams = config['OtherParams']
    cfgModel = config['Models'][args.modelType]

    #random state setup (for repeatability)
    torch.backends.cudnn.deterministic = True #not sure if need this (#ignored if GPU not available)
    torch.manual_seed(cfgParams['seed'])
    torch.cuda.manual_seed_all(cfgParams['seed']) #no need to do this (#ignored if GPU not available)
    # torch.backends.cudnn.benchmark = False #not sure if need this

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    descData, descVocabSize, absVocabSize, absIdx2Word = loadAndPreprocessData.getData(inputTextFile=args.inputTextFile, cpc_codes=cfgParams['cpcCodes'], logger=logger)
    model = eval('models.'+args.modelType)(descVocabSize=descVocabSize, absVocabSize=absVocabSize, 
                                beamSize=args.beamSize, embMult=cfgModel['embMult'], predMaxLen=cfgModel['predMaxLen'], 
                                encMaxLen=cfgModel['encMaxLen'], pad_token=cfgParams['padToken'], 
                                hiddenDim=args.hiddenDim, numLayers=args.numLayers, dropout=args.dropout,
                                numHeads=args.numHeads, decNumLayers=args.decNumLayers)

    #load model
    model, step, metricVal = utils.loadModel(model, f'{args.loadModelName}', device, return_step=True)
    logger.debug(f'Loaded {args.loadModelName} model for {model.__class__.__name__}, which is from step {step} and metric value is {metricVal:.3f}')

    #evaluate
    logger.debug('Starting model inference for the loaded model...')
    modelInference(model=model, descData=descData, abs_idx2word=absIdx2Word, device=device)
    # utils.profileModel(model, val_data, devName='cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug('\n******************************************************************************\n')

