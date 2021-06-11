"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    This is used for testing models
"""

PARENT_DIR = './'
CONFIGFILE = PARENT_DIR + 'config.yaml'

import pytest
import torch
import sys
sys.path.append('./src')
import models
import utils
# from src import models
# from ..ModelBuilding.src import models

def generate_data(batch_size, max_seq_len, vocab_size, seed):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True #ignored if GPU not available
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #ignored if GPU not available

    x = torch.randint(1, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int32)
    max_l = 0
    for b in range(batch_size):
        l = torch.randint(1, max_seq_len, size=(1,)).item()
        x[b,l:] = 0
        max_l = max(l,max_l)
    return x, max_l

def build_model(config, device):
    hiddenDim = config['Models']['Seq2SeqwithXfmrMemEfficient']['hiddenDim']
    numHeads = config['Models']['Seq2SeqwithXfmrMemEfficient']['numHeads']
    embMult = config['Models']['Seq2SeqwithXfmrMemEfficient']['embMult']
    dropout = config['Models']['Seq2SeqwithXfmrMemEfficient']['dropout']
    numLayers = config['Models']['Seq2SeqwithXfmrMemEfficient']['numLayers']
    decNumLayers = config['Models']['Seq2SeqwithXfmrMemEfficient']['decNumLayers']
    predMaxLen = config['Models']['Seq2SeqwithXfmrMemEfficient']['predMaxLen']
    encMaxLen = config['Models']['Seq2SeqwithXfmrMemEfficient']['encMaxLen']
    beamSize = config['OtherParams']['beamSize']
    pad_token = config['OtherParams']['padToken']
    vocab_size = config['OtherParams']['vocabSize']
    xfmrModel = models.Seq2SeqwithXfmrMemEfficient(vocab_size, vocab_size, 
        hiddenDim=hiddenDim, beamSize=beamSize, embMult=embMult, numHeads=numHeads, 
        dropout=dropout, numLayers=numLayers, decNumLayers=decNumLayers, 
        predMaxLen=predMaxLen, encMaxLen=encMaxLen, pad_token=pad_token).to(device)
    return xfmrModel



def test_PositionalEncoding():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = utils.read_params(CONFIGFILE)['Unit_Tests']
    seed = config['OtherParams']['seed']
    batch_size, vocab_size = config['OtherParams']['batchSize'], config['OtherParams']['vocabSize']
    config = config['Models']['Seq2SeqwithXfmrMemEfficient']
    max_seq_len, hidden_dim = config['predMaxLen'], config['hiddenDim']
    x, max_l = generate_data(batch_size, max_seq_len, vocab_size, seed)
    PE = models.PositionalEncoding(hidden_dim, max_seq_len).to(device)
    pred = PE(x.to(device))
    target_shape = [batch_size, max_seq_len, hidden_dim]
    assert list(pred.shape) == target_shape, f'Positional encoding test fails because its output shape is {pred.shape} vs it needs to be {target_shape}'
# test_PositionalEncoding()

def test_XfmrMemEfficientEncoder():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = utils.read_params(CONFIGFILE)['Unit_Tests']
    xfmrModel = build_model(config, device)
    seed = config['OtherParams']['seed']
    batch_size, vocab_size = config['OtherParams']['batchSize'], config['OtherParams']['vocabSize']
    max_seq_len = config['Models']['Seq2SeqwithXfmrMemEfficient']['encMaxLen']
    hidden_dim = config['Models']['Seq2SeqwithXfmrMemEfficient']['hiddenDim']

    x, max_l = generate_data(batch_size, max_seq_len, vocab_size, seed)    
    h,_,_ = xfmrModel.runEncoder(x.to(device))
    targetLen = config['Models']['Seq2SeqwithXfmrMemEfficient']['targetLen']
    assert list(h.shape) == [targetLen, batch_size, hidden_dim], f'The encoder output shape is not correct. It is {list(h.shape)} but needs to be {[targetLen, batch_size, hidden_dim]}'
# test_XfmrMemEfficientEncoder()

def test_XfmrMemEfficientFwdMethod():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = utils.read_params(CONFIGFILE)['Unit_Tests']
    xfmrModel = build_model(config, device)
    seed = config['OtherParams']['seed']
    batch_size, vocab_size = config['OtherParams']['batchSize'], config['OtherParams']['vocabSize']
    max_seq_lenx = config['Models']['Seq2SeqwithXfmrMemEfficient']['encMaxLen']
    max_seq_leny = config['Models']['Seq2SeqwithXfmrMemEfficient']['predMaxLen']
    hidden_dim = config['Models']['Seq2SeqwithXfmrMemEfficient']['hiddenDim']

    x, max_l = generate_data(batch_size, max_seq_lenx, vocab_size, seed)    
    y, max_l = generate_data(batch_size, max_seq_leny, vocab_size, seed)    

    out = xfmrModel(x.to(device),y.to(device))
    assert list(out.shape) == [batch_size, vocab_size, max_seq_leny], f'The final hidden tensor\'s shape is not correct. It\'s {list(out.shape)} but needs to be {[batch_size, vocab_size, max_seq_leny]}'
# test_XfmrMemEfficientFwdMethod()


def test_XfmrMemEfficientEval():
    #Warning: this is slow! (for beam search, change beamsize to 3 in config.yaml)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = utils.read_params(CONFIGFILE)['Unit_Tests']
    xfmrModel = build_model(config, device)
    seed = config['OtherParams']['seed']
    batch_size, vocab_size = config['OtherParams']['batchSize'], config['OtherParams']['vocabSize']
    max_seq_len = config['Models']['Seq2SeqwithXfmrMemEfficient']['encMaxLen']
    hidden_dim = config['Models']['Seq2SeqwithXfmrMemEfficient']['hiddenDim']

    x, _ = generate_data(batch_size, max_seq_len, vocab_size, seed)    
    xfmrModel = build_model(config, device)
    out = xfmrModel.evaluate(x.to(device))
    predLen = config['Models']['Seq2SeqwithXfmrMemEfficient']['evalPredLen']
    assert list(out.shape) == [batch_size, predLen], f'The final hidden tensor\'s shape is not correct. It\'s {list(out.shape)} but needs to be {[batch_size, predLen]}'
# test_XfmrMemEfficientEval()