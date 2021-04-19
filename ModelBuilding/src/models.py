"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Contains various models for text summarization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import evaluate



class Seq2SeqwithAttention(nn.Module):
    """
    This is seq2seq model with Attention. Based upon CS224n Assignment 4.
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqwithAttention, self).__init__()
        self.encoder = encoder
        self.h0_proj = nn.Linear(encoder.hidden_dim, encoder.hidden_dim)
        self.decoder = decoder

    def forward(self, x, y):
        '''
        x is of shape batch x desc_seq_len
        y is of shape batch x abs_seq_len
        '''
        device = x.device

        hinit_dec, yEnc = self.encoder(x) #shape of hinit_dec is: L*D x B x H and shape of YEnc is B x Lenc x 2H

        B, N, D, H = hinit_dec.shape[1], self.encoder.num_layers, (2 if self.encoder.bidir else 1), self.encoder.hidden_dim
        hinit_dec = hinit_dec.view(N,D,B,H)
        hinit_dec = hinit_dec[:,0,:,:] #NXBXH (need to do this because encoder is bidirectional LSTM vs Decoder is always unidirectional)
        hinit_dec = hinit_dec.permute(1,0,2) #BxNxH
        hinit_dec = self.h0_proj(hinit_dec) #BxNxH
        hinit_dec = hinit_dec.permute(1,0,2) #NxBxH

        B, N, H = y.shape[0], self.decoder.num_layers * (2 if self.decoder.bidir else 1), self.decoder.hidden_dim
        c0 = torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)

        use_teacher_forcing=True
        if use_teacher_forcing:
            y_hat, _ = self.decoder(y, (hinit_dec, c0), yEnc)

        return y_hat

    def evaluate(self, x):
        '''
        x is of shape batch x desc_seq_len
        '''
        device = x.device
        hinit_dec, yEnc = self.encoder(x) #shape of hinit_dec is: L*D x B x H and shape of YEnc is B x Lenc x 2H

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
            pred = evaluate.eval_single_batch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], self.decoder, yEnc=yEnc[b:b+1,:,:], predMaxLen=175)
            # pred = evaluate.eval_single_batch_beamsearch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], self.decoder, yEnc=yEnc[b:b+1,:,:], beamSize=3, predMaxLen=175)
            predictions.append(pred)

        #pad predictions
        # predictions, lens = nn.utils.rnn.pad_packed_sequence(predictions, batch_first=True, padding_value=self.decoder.pad_token) #BxL
        max_len = max([p.shape[1] for p in predictions])
        predictions_pad = torch.zeros(size=(B,max_len), dtype=torch.int32, device=device) #BxL
        for b in range(B):
            n = predictions[b].shape[1]
            predictions_pad[b,:n] = predictions[b]
        return predictions_pad


class DecoderLSTMwithAttention(nn.Module):
    """
    This is the LSTM decoder with Attention Layer
    """
    def __init__(self, vocab_size, hidden_dim, dropout=0.0, num_layers=2):
        super(DecoderLSTMwithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = False #always false for decoder
        self.pad_token = 0
        encBidir = True
        encNumDir = 2 if encBidir else 1

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_token)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers, 
                    bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidir)
        self.attenProj = nn.Linear(encNumDir*hidden_dim, hidden_dim, bias=False) #learnable similarity computation
        self.combinedProj = nn.Linear((encNumDir+1)*hidden_dim, hidden_dim)
        self.outProj = nn.Linear(hidden_dim, vocab_size)
        self.dropoutLayer = nn.Dropout(p=self.dropout)



    def forward(self, x, h0c0, yEnc):
        '''
        x: int32 tensor of shape BxL (L == Ldec)
        h0/c0: float32 tensor of shape num_lyrs(N) x batch_size(B) x hidden_dim(H)
        yEnc: float32 tensor of shape B x Lenc x encNumDir*H where encNumDir=2 == B x Lenc x 2H
        '''
        h0,c0 = h0c0
        device = x.device
        lens = (x != self.pad_token).sum(axis=1).to(torch.device('cpu')) #lens for pack_padded_seq need to be on cpu
        x = self.embedding(x)
        x = F.relu(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, (hn,cn) = self.LSTM(x, (h0,c0))
        y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=self.pad_token) #BxLxH

        #steps for computing attention over the encoder
        yEncPrj = self.attenProj(yEnc) #B x Lenc x H
        yEncPrj = yEncPrj.permute(0,2,1) #B x H x Lenc
        simMat = torch.bmm(y, yEncPrj) #B x L x Lenc
        simMat = F.softmax(simMat, dim=2) #B x L x Lenc
        alpha = torch.bmm(simMat, yEnc) #B x L x 2H
        y = torch.cat((y,alpha), dim=2) #B x L x 3H
        y = self.combinedProj(y) #B x L x H
        y = torch.tanh(y) #didn't see much difference with relu() here
        y = self.dropoutLayer(y) #BxLxH

        #project to the vocab dimension
        y = self.outProj(y) #BxLxVocab_size(V)
        y = y.permute(0,2,1) #BxVxL (need to do this for crossentropy loss). No need to do softmax as cross entropy loss does it.
        return y, (hn,cn)




class EncoderLSTMwithAttention(nn.Module):
    """
    This is the LSTM encoder for use with Attention base Decoder
    """
    def __init__(self, vocab_size, hidden_dim, dropout=0.0, num_layers=2, bidir=True):
        super(EncoderLSTMwithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = bidir
        self.pad_token = 0

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_token)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers, 
                    bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidir)

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
        y,(h,c) = self.LSTM(x, (h0,c0)) #h: numLyrs*numDirsxBxH
        y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=self.pad_token) #BxLxnumDir*H
        return h, y



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
            pred = evaluate.eval_single_batch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], self.decoder, predMaxLen=175)
            # pred = evaluate.eval_single_batch_beamsearch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], self.decoder, beamSize=3, predMaxLen=175)
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
    def __init__(self, vocab_size, hidden_dim, dropout=0.0, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = False #always false
        self.pad_token = 0

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_token)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers, 
                    bias=True, batch_first=True, dropout=dropout, bidirectional=self.bidir)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)


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
    def __init__(self, vocab_size, hidden_dim, dropout=0.0, num_layers=2, bidir=True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidir = bidir
        self.pad_token = 0

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_token)
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

