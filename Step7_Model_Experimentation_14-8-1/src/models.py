"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Contains various models for text summarization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import utils



class Seq2Seq(nn.Module):
    """
    This is basic seq2seq model 
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.h0_proj = nn.Linear(encoder.hidden_dim, encoder.hidden_dim)
        self.decoder = decoder

    def forward(self, x, y, use_teacher_forcing=True):
        '''
        x is of shape batch x desc_seq_len
        y is of shape batch x abs_seq_len
        '''
        device = x.device

        hinit_dec = self.encoder(x) #shape of hinit_dec is: L*D x B x H

        B, N, D, H = hinit_dec.shape[1], self.encoder.num_layers, (2 if self.encoder.bidir else 1), self.encoder.hidden_dim
        hinit_dec = hinit_dec.view(N,D,B,H)
        hinit_dec = hinit_dec[:,0,:,:] #NXBXH (need to do this because encoder is bidirectional LSTM vs Decoder is always unidirectional)
        hinit_dec = hinit_dec.permute(1,0,2) #BxNxH
        hinit_dec = self.h0_proj(hinit_dec) #BxNxH
        hinit_dec = hinit_dec.permute(1,0,2) #NxBxH

        B, N, H = y.shape[0], self.decoder.num_layers * (2 if self.decoder.bidir else 1), self.decoder.hidden_dim
        c0 = torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)

        if use_teacher_forcing:
            y_hat, _ = self.decoder(y, (hinit_dec, c0))
        return y_hat
    


    def evaluate(self, x):
        '''
        x is of shape batch x desc_seq_len
        '''
        pred_max_len = 175
        def eval_single_batch_helper(h0, c0):
            start_token = 3*torch.ones(size=(1,1), dtype=torch.int32, device=device)
            stop_token = 4*torch.ones(size=(1,1), dtype=torch.int32, device=device)
            pad_token = 0*torch.ones(size=(1,1), dtype=torch.int32, device=device)
            hprev, cprev = h0, c0
            yprev = start_token
            predictions = []
            while yprev != stop_token and yprev != pad_token and len(predictions) < pred_max_len:
                # yprev: 1x1, ynxt: 1xVx1, hprev/hnxt: Nx1xH, cprev/cnxt: Nx1xH
                try:
                    ynxt, (hnxt, cnxt) = self.decoder(yprev, (hprev, cprev))
                except:
                    print(yprev)
                hprev, cprev = hnxt, cnxt
                yprev = torch.argmax(ynxt, axis=1).to(dtype=torch.int32, device=device)
                predictions.append(yprev)
            return torch.cat(predictions, dim=1) #1xL

        device = x.device
        hinit_dec = self.encoder(x) #shape of hinit_dec is: L*D x B x H

        B, N, D, H = hinit_dec.shape[1], self.encoder.num_layers, (2 if self.encoder.bidir else 1), self.encoder.hidden_dim
        hinit_dec = hinit_dec.view(N,D,B,H)
        hinit_dec = hinit_dec[:,0,:,:] #NXBXH (need to do this because encoder is bidirectional LSTM vs Decoder is always unidirectional)
        hinit_dec = hinit_dec.permute(1,0,2) #BxNxH
        hinit_dec = self.h0_proj(hinit_dec) #BxNxH
        hinit_dec = hinit_dec.permute(1,0,2) #NxBxH

        N, H = self.decoder.num_layers * (2 if self.decoder.bidir else 1), self.decoder.hidden_dim
        c0 = torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)

        predictions = []
        for b in range(B):
            pred = eval_single_batch_helper(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:])
            predictions.append(pred)

        #pad predictions
        # predictions, lens = nn.utils.rnn.pad_packed_sequence(predictions, batch_first=True, padding_value=self.decoder.pad_token) #BxL
        max_len = max([p.shape[1] for p in predictions])
        predictions_pad = torch.zeros(size=(B,max_len), dtype=torch.int32, device=device) #BxL
        for b in range(B):
            n = predictions[b].shape[1]
            predictions_pad[b,:n] = predictions[b]
        return predictions_pad


class DecoderLSTM(nn.Module):
    """
    This is the LSTM decoder
    """
    def __init__(self, embed_size, hidden_dim, dropout=0.0, num_layers=2, bidir=True):
        super(DecoderLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = bidir
        self.pad_token = 0

        self.embedding = nn.Embedding(embed_size, hidden_dim, padding_idx=self.pad_token)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers, 
                    bias=True, batch_first=True, dropout=dropout, bidirectional=self.bidir)
        self.out_proj = nn.Linear(hidden_dim, embed_size)


    def forward(self, x, h0c0):
        '''
        x: int32 tensor of shape BxL
        h0: float32 tensor of shape num_lyrs(N) x batch_size(B) x hidden_dim(H)
        '''
        h0,c0 = h0c0
        device = x.device
        lens = (x != self.pad_token).sum(axis=1).to(torch.device('cpu')) #lens for pack_padded_seq need to be on cpu
        x = self.embedding(x)
        x = F.relu(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, (hn,cn) = self.LSTM(x, (h0,c0))
        y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=self.pad_token) #BxLxH
        y = self.out_proj(y) #BxLxVocab_size(V)
        y = y.permute(0,2,1) #BxVxL (need to do this for crossentropy loss). No need to do softmax as cross entropy loss does it.
        return y, (hn,cn)




class EncoderLSTM(nn.Module):
    """
    This is the LSTM encoder
    """
    def __init__(self, embed_size, hidden_dim, dropout=0.0, num_layers=2, bidir=True):
        super(EncoderLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = bidir
        self.pad_token = 0

        self.embedding = nn.Embedding(embed_size, hidden_dim, padding_idx=self.pad_token)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers, 
                    bias=True, batch_first=True, dropout=dropout, bidirectional=self.bidir)

    def forward(self, x):
        '''
        x: int32 tensor of shape BxL 
        '''
        device = x.device

        lens = (x != self.pad_token).sum(axis=1).to(torch.device('cpu')) #lens for pack_padded_seq need to be on cpu
        B, N, H = x.shape[0], self.num_layers * (2 if self.bidir else 1), self.hidden_dim
        x = self.embedding(x)
        x = F.relu(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        h0,c0 = torch.zeros(N,B,H).to(device), torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)
        y,(h,c) = self.LSTM(x, (h0,c0))
        # y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=self.pad_token)
        return h

