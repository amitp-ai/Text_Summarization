"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    This is used for testing the training code and different utility methods and classes
"""


PARENT_DIR = './'
CONFIGFILE = PARENT_DIR + 'config.yaml'

import pytest
import torch
# from ..ModelBuilding.src import models
import sys
sys.path.append(f'{PARENT_DIR}src')
import train
import utils
import test_models

def test_dataCollection():
    """This is slow to run!"""
    logger = utils.create_logger('test.log', logDir='./test')
    config = utils.read_params(CONFIGFILE)['Unit_Tests']['OtherParams']
    train_data, val_data, lang_train = utils.get_data(use_full_vocab=config['fullVocab'], 
            cpc_codes=config['cpcCodes'], fname=config['fname'], 
            train_size=config['trainSize'], val_size=config['valSize'], logger=logger)
    assert ((len(train_data) == config['trainSize']) and (len(val_data) == config['valSize'])), f"Train data size is {len(train_data)} and needs to be {config['trainSize']} & val data size {len(val_data)} and needs to be {config['valSize']}"
    return train_data, val_data, lang_train
# test_dataCollection()

def test_training():
    """This is method slow to run!
    """
    logger = utils.create_logger('test.log', logDir='./test')

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = utils.read_params(CONFIGFILE)['Unit_Tests']
    xfmrModel = test_models.build_model(config, device)

    seed = config['OtherParams']['seed']
    batch_size, vocab_size = config['OtherParams']['batchSize'], config['OtherParams']['vocabSize']
    max_seq_len = config['Models']['Seq2SeqwithXfmrMemEfficient']['encMaxLen']
    hidden_dim = config['Models']['Seq2SeqwithXfmrMemEfficient']['hiddenDim']

    logger.debug("Testing runEncoder...")
    x, max_l = test_models.generate_data(batch_size, max_seq_len, vocab_size, seed)    
    h,_,_ = xfmrModel.runEncoder(x.to(device))

    logger.debug('Testing the training loop...')
    config = config['OtherParams']
    train_data, val_data, lang_train = utils.get_data(use_full_vocab=config['fullVocab'], 
            cpc_codes=config['cpcCodes'], fname=config['fname'], 
            train_size=config['trainSize'], val_size=config['valSize'], logger=logger)

    #load model
    model, step, metricVal = utils.loadModel(xfmrModel, f"{config['loadModelName']}", device, return_step=True)

    #train model
    # out = xfmrModel(x.to(device),y.to(device))
    train.train(model=model, train_data=train_data, val_data=val_data, abs_idx2word=lang_train.abs_idx2word, 
        device=device, batch_size=config['batchSize'], num_epochs=config['numEpochs'], lr=config['lr'], 
        print_every_iters=config['printEveryIters'], savedModelBaseName=config['savedModelBaseName'], 
        step=step, bestMetricVal=metricVal, l2Reg=config['l2Reg'])

