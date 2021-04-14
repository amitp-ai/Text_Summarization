import pytest
import torch
import sys
sys.path.append('../ModelBuilding/src')
import models

def get_ModelMetaData(): 
    #could make this into a fixture, but not necessary
    batch_size, max_seq_len = 2, 20
    vocab_size, hidden_dim = 10, 5
    return (batch_size, max_seq_len, vocab_size, hidden_dim)

def generate_data(x):
    (batch_size, max_seq_len, vocab_size, hidden_dim) = get_ModelMetaData()
    max_l = 0
    for b in range(batch_size):
        l = torch.randint(1, max_seq_len, size=(1,)).item()
        x[b,l:] = 0
        max_l = max(l,max_l)
    return x, max_l

def test_EncLSTM():
    (batch_size, max_seq_len, vocab_size, hidden_dim) = get_ModelMetaData()
    x = torch.randint(1, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int32)
    x, max_l = generate_data(x)    
    enc = models.EncoderLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2, bidir=True)
    h = enc(x)
    assert list(h.shape) == [enc.num_layers * (2 if enc.bidir else 1), batch_size, hidden_dim], 'The final hidden tensor\'s shape is not correct'

def test_DecLSTM():
    (batch_size, max_seq_len, vocab_size, hidden_dim) = get_ModelMetaData()
    x = torch.randint(1, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int32)
    x, max_l = generate_data(x)    
    dec = models.DecoderLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2, bidir=False)
    h0 = torch.zeros(size=(dec.num_layers * (2 if dec.bidir else 1), batch_size, hidden_dim))
    c0 = h0.clone()
    y, _ = dec(x, (h0, c0))
    assert list(y.shape) == [batch_size, dec.vocab_size, max_l], 'The output tensor\'s shape is not correct'

def test_Seq2Seq():
    (batch_size, max_seq_len, vocab_size, hidden_dim) = get_ModelMetaData()
    x = torch.randint(1, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int32)
    x, _ = generate_data(x)    
    y = torch.randint(1, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int32)
    y, max_l = generate_data(y)    

    enc = models.EncoderLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2, bidir=True)
    dec = models.DecoderLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2, bidir=False)
    model = models.Seq2Seq(enc, dec)
    yhat = model(x,y)
    assert list(yhat.shape) == [batch_size, dec.vocab_size, max_l], 'The output tensor\'s shape is not correct'
