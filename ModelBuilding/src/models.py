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
import numpy as np


def SummaryGeneration(hinit_dec, c0, decoder, predMaxLen, yEnc=None, encMask=None, beamSize=0):
    """ This method sequentially generates the summary """
    predictions = []
    B = hinit_dec.shape[1]
    for b in range(B):
        if yEnc is not None: 
            yEncTmp = yEnc[b:b+1,:,:]
            encMaskTmp = encMask[b:b+1,:]
        else: 
            yEncTmp = yEnc
            encMaskTmp = encMask 
        if beamSize == 0:
            pred = evaluate.eval_single_batch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], decoder, yEnc=yEncTmp, 
                                            encMask=encMaskTmp, predMaxLen=predMaxLen)
        else:
            pred = evaluate.eval_single_batch_beamsearch(hinit_dec[:,b:b+1,:], c0[:,b:b+1,:], decoder, yEnc=yEncTmp, 
                                                        encMask=encMaskTmp, beamSize=beamSize, predMaxLen=predMaxLen)
        predictions.append(pred)
    
    #pad predictions (inorder to convert into a tensor)
    # predictions, lens = nn.utils.rnn.pad_packed_sequence(predictions, batch_first=True, padding_value=decoder.pad_token) #BxL
    max_len = max([p.shape[1] for p in predictions])
    predictions_pad = torch.zeros(size=(B,max_len), dtype=torch.int32, device=hinit_dec.device) #BxL
    for b in range(B):
        n = predictions[b].shape[1]
        predictions_pad[b,:n] = predictions[b]
    return predictions_pad

def noTeacherForcing(B, L, decoder, hinit_dec, c0, yEnc=None, encMask=None):
    ''' For doing no teacher forcing when training 
    B: batch size, L: seq length, decoder: decoder, hinit_dec: initial value for h, c0: initial value for c,
    yEnc: RNN output from encoder (used for attention)
    '''
    hprev, cprev = hinit_dec, c0
    start_token = 3*torch.ones(size=(B,1), dtype=torch.int32, device=device)
    yprev = start_token
    predictions = []
    # y_hat = torch.zeros(size=(B,decoder.vocab_size,L), dtype=torch.float32, device=device)
    for t in range(1,L):
        if yEnc and encMask: ynxt, (hnxt, cnxt) = decoder(yprev, (hprev, cprev), yEnc, encMask) #ynxt: BxVx1
        else: ynxt, (hnxt, cnxt) = decoder(yprev, (hprev, cprev)) #ynxt: BxVx1
        hprev, cprev = hnxt, cnxt
        yprev = torch.argmax(ynxt, dim=1).to(dtype=torch.int32) #Bx1
        yprev = torch.where(yprev == 0, start_token[0,0], yprev) #replace pad with start token and feed it as input for next time step
        predictions.append(ynxt)
        # y_hat[:,:,t:t+1] = ynxt #the model does not seem to learn much with this approach (loss decrease but rouge 1 stays at 0)
    y_hat = torch.cat(predictions, dim=1) #BxVxL
    del predictions
    return y_hat


class Seq2SeqwithAttention(nn.Module):
    """
    This is seq2seq model with Attention. Based upon CS224n Assignment 4.
    """
    def __init__(self, descVocabSize, absVocabSize, hiddenDim, numLayers, dropout, bidir=True, tfThresh=0.0, beamSize=0):
        super(Seq2SeqwithAttention, self).__init__()
        self.encoder = EncoderLSTMwithAttention(vocab_size=descVocabSize, hidden_dim=hiddenDim, 
                                                num_layers=numLayers, bidir=bidir, dropout=dropout)
        self.h0_proj = nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim)
        self.decoder = DecoderLSTMwithAttention(vocab_size=absVocabSize, hidden_dim=hiddenDim, 
                                                num_layers=numLayers, dropout=dropout)
        self.predMaxLen = 175
        self.tfThresh = tfThresh
        self.beamSize = beamSize

    def forward(self, x, y):
        '''
        x is of shape batch x desc_seq_len
        y is of shape batch x abs_seq_len
        '''
        device = x.device
        encMask = (x != self.encoder.pad_token).to(torch.uint8) #batch x desc_seq_len

        hinit_dec, yEnc = self.encoder(x) #shape of hinit_dec is: L*D x B x H and shape of YEnc is B x Lenc x 2H

        B, N, D, H = hinit_dec.shape[1], self.encoder.num_layers, (2 if self.encoder.bidir else 1), self.encoder.hidden_dim
        hinit_dec = hinit_dec.view(N,D,B,H)
        hinit_dec = hinit_dec[:,0,:,:] #NXBXH (need to do this because encoder is bidirectional LSTM vs Decoder is always unidirectional)
        hinit_dec = hinit_dec.permute(1,0,2) #BxNxH
        hinit_dec = self.h0_proj(hinit_dec) #BxNxH
        hinit_dec = hinit_dec.permute(1,0,2) #NxBxH

        B, N, H = y.shape[0], self.decoder.num_layers * (2 if self.decoder.bidir else 1), self.decoder.hidden_dim
        c0 = torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)

        useTeacherForcing=True
        if np.random.rand() < self.tfThresh: #X% of the time no teacher forcing (i.e. use previous word prediction as next time step's input)
            useTeacherForcing = False
        if useTeacherForcing:
            y_hat, _ = self.decoder(y, (hinit_dec, c0), yEnc, encMask) #BxVxL
        else:
            L = y.shape[1]
            y_hat = noTeacherForcing(B, L, self.decoder, hinit_dec, c0, yEnc, encMask)
        return y_hat


    def evaluate(self, x):
        '''
        x is of shape batch x desc_seq_len
        '''
        device = x.device
        encMask = (x != self.encoder.pad_token).to(torch.uint8) #batch x desc_seq_len
        hinit_dec, yEnc = self.encoder(x) #shape of hinit_dec is: L*D x B x H and shape of YEnc is B x Lenc x 2H

        B, N, D, H = hinit_dec.shape[1], self.encoder.num_layers, (2 if self.encoder.bidir else 1), self.encoder.hidden_dim
        hinit_dec = hinit_dec.view(N,D,B,H)
        hinit_dec = hinit_dec[:,0,:,:] #NXBXH (need to do this because encoder is bidirectional LSTM vs Decoder is always unidirectional)
        hinit_dec = hinit_dec.permute(1,0,2) #BxNxH
        hinit_dec = self.h0_proj(hinit_dec) #BxNxH
        hinit_dec = hinit_dec.permute(1,0,2) #NxBxH

        N, H = self.decoder.num_layers * (2 if self.decoder.bidir else 1), self.decoder.hidden_dim
        c0 = torch.zeros(N,B,H).to(device) #B is second term even though have batch_first=True (which only applies to input)

        predictions = SummaryGeneration(hinit_dec, c0, self.decoder, yEnc=yEnc, encMask=encMask,
                                    predMaxLen=self.predMaxLen, beamSize=self.beamSize)
        return predictions


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

    def forward(self, x, h0c0, yEnc, encMask):
        '''
        x: int32 tensor of shape BxL (L == Ldec)
        h0/c0: float32 tensor of shape num_lyrs(N) x batch_size(B) x hidden_dim(H)
        yEnc: float32 tensor of shape B x Lenc x encNumDir*H where encNumDir=2 == B x Lenc x 2H
        encMask: B x Lenc
        '''
        device = x.device
        decMask = (x != self.pad_token).to(torch.uint8).unsqueeze(dim=2) #B x Ldec x 1

        h0,c0 = h0c0
        lens = (x != self.pad_token).sum(axis=1).to(torch.device('cpu')) #lens for pack_padded_seq need to be on cpu
        x = self.embedding(x)
        x = F.relu(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, (hn,cn) = self.LSTM(x, (h0,c0))
        y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, padding_value=self.pad_token) #BxLxH
        decMask = decMask[:,:y.shape[1],:] #bcse the length may be reduced after pad_packed_seq()

        #computing a multiplicative attention (https://ruder.io/deep-learning-nlp-best-practices/index.html#attention)
        yEncPrj = self.attenProj(yEnc) #B x Lenc x H
        yEncPrj = yEncPrj.permute(0,2,1) #B x H x Lenc
        simMat = torch.bmm(y, yEncPrj) #B x L x Lenc

        encMask = encMask[:,:yEnc.shape[1]] #bcse the length may be reduced after pad_packed_seq() in encoder layer
        simMat = maskedSoftmax(simMat, encMask, dim=2) #B x L x Lenc
        simMat = simMat * decMask #B x Ldec x Lenc
        alpha = torch.bmm(simMat, yEnc) #B x L x 2H
        y = torch.cat((y,alpha), dim=2) #B x L x 3H
        y = self.combinedProj(y) #B x L x H
        y = torch.tanh(y) #didn't see much difference with relu() here
        y = self.dropoutLayer(y) #BxLxH

        #project to the vocab dimension
        y = self.outProj(y) #BxLxVocab_size(V)
        y = y.permute(0,2,1) #BxVxL (need to do this for crossentropy loss). No need to do softmax as cross entropy loss does it.
        return y, (hn,cn)


def maskedSoftmax(simMat, encMask, dim=2):
    '''
    simMat: Batch x decLen x encLen
    encMask: Batch x encLen (0 where it is 0 i.e. pad token and 1 otherwise)
    '''
    largeNegNum = -1e30
    encMask = encMask.unsqueeze(dim=1) #batch x 1 x encLen
    simMat = (encMask)*simMat + (1-encMask)*largeNegNum
    simMat = F.softmax(simMat, dim=2) #batch x decLen x encLen
    return simMat


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
    def __init__(self, descVocabSize, absVocabSize, hiddenDim, numLayers, dropout, bidir=True, tfThresh=0.0, beamSize=0):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderLSTM(vocab_size=descVocabSize, hidden_dim=hiddenDim, 
                                  num_layers=numLayers, bidir=bidir, dropout=dropout)
        self.h0_proj = nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim)
        self.decoder = DecoderLSTM(vocab_size=absVocabSize, hidden_dim=hiddenDim, 
                                  num_layers=numLayers, dropout=dropout)
        self.predMaxLen = 175
        self.tfThresh = tfThresh
        self.beamSize = beamSize


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

        predictions = SummaryGeneration(hinit_dec, c0, self.decoder, predMaxLen=self.predMaxLen)
        return predictions

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
        return h #(N,B,H)

############## Transformer Based Model ###########################


class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen=5000):
        '''
        Positional encoding to be used with transformer layer.
        The output is sinusoidal with respect to the position (so position can be thought of a time).
        And the frequency of this sinusoid is different for each element in the dModel dimension, so we end up with 
        dModel number of sinusoids. The frequencies have geometric progression.
        
        dModel <int>: size of embedding dimension
        '''
        super(PositionalEncoding, self).__init__()
        hpar = 10000.0
        pe = torch.zeros(maxLen, dModel) #maxLen x d
        position = torch.arange(0, maxLen, dtype=torch.float32).unsqueeze(1) #maxLen x 1
        div_term = hpar ** (2*torch.arange(0,dModel,2)/dModel).to(dtype=torch.float32).unsqueeze(0) #1 x d/2

        pe[:, 0::2] = torch.sin(position / div_term) #maxLen x d/2
        pe[:, 1::2] = torch.cos(position / div_term) #maxLen x d/2
        pe = pe.unsqueeze(0)  #1 x maxLen x d
        self.register_buffer('pe', pe) #not trainable but saved with the model and moved to cpu/gpu with model.to()

    def forward(self, x):
        '''x: B x Len'''
        x = torch.zeros_like(x).unsqueeze(-1) + self.pe[:, :x.size(1), :] #B x len x d
        return x


class Seq2SeqwithXfmr(nn.Module):
    """
    This is seq2seq model using Transformer network.
    """
    def __init__(self, descVocabSize, absVocabSize, hiddenDim=200, numLayers=6, dropout=0.0, tfThresh=0.0, beamSize=0):
        super(Seq2SeqwithXfmr, self).__init__()
        self.predMaxLen = 150
        self.encMaxLen = 4000
        self.pad_token = 0
        self.numHeads=4
        self.tfThresh = tfThresh
        self.beamSize = beamSize
        self.hidDim = hiddenDim
        self.embDim = hiddenDim
        self.encEmbedding = nn.Embedding(descVocabSize, self.embDim, padding_idx=self.pad_token)
        self.encPos = PositionalEncoding(dModel=self.embDim, maxLen=self.encMaxLen)
        self.decEmbedding = nn.Embedding(absVocabSize, self.embDim, padding_idx=self.pad_token)
        self.decPos = PositionalEncoding(dModel=self.embDim, maxLen=self.predMaxLen)
        self.xFmrLyr = nn.Transformer(d_model=hiddenDim, nhead=self.numHeads, num_encoder_layers=numLayers, 
                                    num_decoder_layers=numLayers, dim_feedforward=hiddenDim, 
                                    dropout=dropout, activation='relu')
        self.decoder = self.xFmrLyr.decoder #for use by the evaluation menthods
        self.outPrj = nn.Linear(hiddenDim, absVocabSize)
        # self._resetParams() #not needed as the transformer layer does so inside it
        # utils.getModelInfo(self)

    def _resetParams(self):
        for i, p in enumerate(self.parameters()):
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def getCausalityMask(self, sz):
        '''
        Based upon: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        generate_square_subsequent_mask(sz)
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        largeNegNum = float('-inf')
        mask = mask.float().masked_fill(mask == 0, largeNegNum).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        '''
        x is of shape batch x Ldesc
        y is of shape batch x Labs
        '''
        device = x.device

        xMask = (x != self.pad_token).unsqueeze(-1) #B x Ldesc x 1
        x = self.encEmbedding(x) + self.encPos(x) #B x Lenc x E
        x = x * xMask #zero out all the inputs corresponding to pad tokens
        x = x.transpose(0,1) #Lenc x B x E

        yMask = (y != self.pad_token).unsqueeze(-1) #B x Labs x 1
        y = self.decEmbedding(y) + self.decPos(y) #B x Labs x E
        y = y * yMask #zero out all the inputs corresponding to pad tokens
        y = y.transpose(0,1)  #Ldec x B x E

        '''
        Transformer dimensions:
            src: (S, B, E), tgt: (T, B, E), src_mask: (S, S), tgt_mask: (T, T), memory_mask: (T, S), src_key_padding_mask: (B, S), tgt_key_padding_mask: (B, T), memory_key_padding_mask: (B, S)
        '''
        Lenc, B, _ = x.shape
        Ldec = y.shape[0]
        srcMask = None #bcse causality does not apply (so attend to everything)
        tgtMask = self.getCausalityMask(Ldec).to(device=device) #causality mask for the decoder
        memMask = None #bcse causality does not apply (so attend to everything)
        srcKeyPadMask = xMask.logical_not().squeeze(-1) #B x Ldesc
        tgtKeyPadMask = yMask.logical_not().squeeze(-1) #B x Labs
        memKeyPadMask = srcKeyPadMask #.clone().detach() #B x Ldesc

        # output = self.xFmrLyr(src=x, tgt=y)
        output = self.xFmrLyr(src=x, tgt=y, src_mask=srcMask, tgt_mask=tgtMask, memory_mask=memMask, src_key_padding_mask=srcKeyPadMask, tgt_key_padding_mask=tgtKeyPadMask, memory_key_padding_mask=memKeyPadMask) #Ldec x B x H
        output = output.transpose(0,1) #B x Ldec x H
        output = self.outPrj(output) #B x Ldec x V
        output = output.transpose(1,2) #B x V x Ldec  (need to do this for crossentropy loss. But no need to do softmax as cross entropy loss does it.)
        return output


    def evaluate(self, x):
        '''
        x is of shape batch x Lenc.
        This function can be optimized by only calling the decoder inside the transformer and not the encoder every time.

        Transformer dimensions:
            src: (S, B, E), tgt: (T, B, E), src_mask: (S, S), tgt_mask: (T, T), memory_mask: (T, S), src_key_padding_mask: (B, S), tgt_key_padding_mask: (B, T), memory_key_padding_mask: (B, S)
        '''
        device = x.device

        xMask = (x != self.pad_token).unsqueeze(-1) #B x Ldesc x 1
        x = self.encEmbedding(x) + self.encPos(x) #B x Lenc x E
        x = x * xMask #zero out all the inputs corresponding to pad tokens
        x = x.transpose(0,1) #Lenc x B x E
        Lenc, B, _ = x.shape
        srcMask = None #bcse causality does not apply (so attend to everything)
        memMask = None #bcse causality does not apply (so attend to everything)
        srcKeyPadMask = xMask.logical_not().squeeze(-1) #B x Ldesc
        memKeyPadMask = srcKeyPadMask #.clone().detach() #B x Ldesc
        memory = self.xFmrLyr.encoder(x, mask=srcMask, src_key_padding_mask=srcKeyPadMask) #Lenc x B x E

        predictions = []
        for b in range(B):
            if self.beamSize == 0:
                prediction = evaluate.evaluateXfmrSingleBatch(self, memory[:,b:b+1,:], memMask, memKeyPadMask[b:b+1,:])
            else:
                prediction = evaluate.evaluateXfmrSingleBatchBeamSearch(self, memory[:,b:b+1,:], memMask, memKeyPadMask[b:b+1,:], beamSize=self.beamSize)
            predictions.append(prediction)
        #pad predictions (inorder to convert into a tensor)
        # predictions, lens = nn.utils.rnn.pad_packed_sequence(predictions, batch_first=True, padding_value=decoder.pad_token) #BxL
        max_len = max([p.shape[1] for p in predictions])
        predictions_pad = torch.zeros(size=(B,max_len), dtype=torch.int32, device=device) #BxLdec
        for b in range(B):
            n = predictions[b].shape[1]
            predictions_pad[b:b+1,:n] = predictions[b]
        return predictions_pad #B x Ldec


class reuseEmbeddings(nn.Module):
    def __init__(self, embeddings):
        super(reuseEmbeddings, self).__init__()
        self.embeddings = embeddings
    
    def forward(self, x):
        """ x is of the shape (B x L x H) 
            self.embeddings.weight is of shape (V x H), where V is the size of vocab
        """
        wEmb = self.embeddings.weight.unsqueeze(0).transpose(1,2) #(1,H,V)
        res = torch.matmul(x, wEmb) #(B,L,V)
        return res


class Seq2SeqwithXfmrMemEfficient(nn.Module):
    """
    This is seq2seq model using Transformer network. It is much more memory efficient as we split the encoder input
     into multiple overlapping chunks.

    As per in the "Attention is all you need paper", the output of the embeddings (for encoder and decoder) is multiplied
    by sqrt(embDim) before summing with posEnc(). This ensures that word embedding representations are weighted more heavily vs
    posEnc().
    """
    def __init__(self, descVocabSize, absVocabSize, hiddenDim=200, numLayers=6, dropout=0.0, tfThresh=0.0, beamSize=0,
                numHeads=4):
        super(Seq2SeqwithXfmrMemEfficient, self).__init__()
        self.predMaxLen = 150
        self.encMaxLen = 4000
        self.pad_token = 0
        self.numHeads=numHeads
        self.tfThresh = tfThresh
        self.beamSize = beamSize
        self.hidDim = hiddenDim
        self.embDim = hiddenDim
        self.embMult = 4 #hiddenDim (hiddenDim seems to work worse)
        self.dropout = dropout
        self.numLayers = numLayers
        self.decNumLayers = numLayers+2
        self.encEmbedding = nn.Embedding(descVocabSize, self.embDim, padding_idx=self.pad_token)
        self.encPos = PositionalEncoding(dModel=self.embDim, maxLen=self.encMaxLen)
        self.decEmbedding = nn.Embedding(absVocabSize, self.embDim, padding_idx=self.pad_token)
        self.decPos = PositionalEncoding(dModel=self.embDim, maxLen=self.predMaxLen)

        encoderLayer = nn.TransformerEncoderLayer(d_model=self.embDim, nhead=self.numHeads, 
                                                dim_feedforward=self.hidDim, dropout=self.dropout, activation='relu')
        encoderNorm = nn.LayerNorm(self.embDim)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=self.numLayers, norm=encoderNorm)

        decoderLayer = nn.TransformerDecoderLayer(d_model=self.embDim, nhead=self.numHeads, 
                                                dim_feedforward=self.hidDim, dropout=self.dropout, activation='relu')
        decoderNorm = nn.LayerNorm(self.embDim)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=self.decNumLayers, norm=decoderNorm)

        weightTying = False #not tying weights performs better (trains faster and has better validation score even for the same training loss)
        if weightTying:
            assert self.embDim == self.hidDim, 'Can only do weight tying if Decoder Embeddings dimension and Hidden dimension are equal'
            self.outPrj = reuseEmbeddings(self.decEmbedding) #(speeds up training) but this only works if embDim == hiddenDim
        else:
            self.outPrj = nn.Linear(self.hidDim, absVocabSize)

        self._resetParameters()
        # utils.getModelInfo(self)

    def _resetParameters(self):
        r"""Initialize all the parameters in the model."""
        for n,p in self.named_parameters():
            if p.dim() > 1:
                if 'Embedding' in n: 
                    nn.init.uniform_(p, -1, 1)
                else: 
                    nn.init.xavier_uniform_(p)


    def getCausalityMask(self, sz):
        '''
        Based upon: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        generate_square_subsequent_mask(sz)
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        largeNegNum = float('-inf')
        mask = mask.float().masked_fill(mask == 0, largeNegNum).masked_fill(mask == 1, float(0.0))
        return mask

    def  runEncoder(self, x):
        ''' x is of shape : B x Lenc'''
        def helper(xsub):
            xMask = (xsub != self.pad_token).unsqueeze(-1) #B x Lenc/n x 1
            xsub = self.encEmbedding(xsub)*np.sqrt(self.embMult) + self.encPos(xsub) #B x Lenc/n x E
            xsub = xsub * xMask #zero out all the inputs corresponding to pad tokens
            xsub= xsub.transpose(0,1) #Lenc/n x B x E

            srcKeyPadMask = xMask.logical_not().squeeze(-1) #B x Lenc/n
            srcKeyPadMask[:,0] = False #assume first position is non pad (otherwise self.encoder() retuns tensor with nan)
            srcMask = None #bcse causality does not apply (so attend to everything)
            memory = self.encoder(xsub, mask=srcMask, src_key_padding_mask=srcKeyPadMask) #Lenc/n x B x E

            return memory, srcKeyPadMask

        numSplits, overLap = 4, 100
        Lenc = x.size(1)
        split = Lenc//numSplits
        memory = []
        memKeyPadMask = []

        for i in range(numSplits):
            strt, stp = i*split, (i+1)*split + overLap
            xTmp = x[:,strt:stp] #B x Lenc/n
            memoryTmp, srcKeyPadMaskTmp = helper(xTmp)
            memory.append(memoryTmp)
            memKeyPadMask.append(srcKeyPadMaskTmp)

        memory = torch.cat(memory, dim=0) #Lenc x B x E
        memKeyPadMask = torch.cat(memKeyPadMask, dim=1) #B x Lenc
        memMask = None #bcse causality does not apply (so attend to everything)

        return memory, memKeyPadMask, memMask

    def forward(self, x, y):
        '''
        x is of shape batch x Ldesc
        y is of shape batch x Labs
        '''

        '''
        Transformer dimensions:
            src: (S, B, E), tgt: (T, B, E), src_mask: (S, S), tgt_mask: (T, T), memory_mask: (T, S), src_key_padding_mask: (B, S), tgt_key_padding_mask: (B, T), memory_key_padding_mask: (B, S)
        '''
        device = x.device

        memory, memKeyPadMask, memMask = self.runEncoder(x) #memory shape is: Lenc x B x E

        yMask = (y != self.pad_token).unsqueeze(-1) #B x Ldec x 1
        y = self.decEmbedding(y)*np.sqrt(self.embMult) + self.decPos(y) #B x Ldec x E
        y = y * yMask #zero out all the inputs corresponding to pad tokens
        y = y.transpose(0,1)  #Ldec x B x E

        Ldec = y.size(0)
        tgtMask = self.getCausalityMask(Ldec).to(device=device) #causality mask for the decoder (Ldec x Ldec)
        tgtKeyPadMask = yMask.logical_not().squeeze(-1) #B x Ldec

        output = self.decoder(y, memory, tgt_mask=tgtMask, memory_mask=memMask,
                        tgt_key_padding_mask=tgtKeyPadMask, memory_key_padding_mask=memKeyPadMask) #Ldec x b x H
        
        output = output.transpose(0,1) #B x Ldec x H
        output = self.outPrj(output) #B x Ldec x V
        output = output.transpose(1,2) #B x V x Ldec  (need to do this for crossentropy loss. But no need to do softmax as cross entropy loss does it.)
        return output

    def evaluate(self, x):
        '''
        x is of shape batch x Lenc.
        This function can be optimized by only calling the decoder inside the transformer and not the encoder every time.

        Transformer dimensions:
            src: (S, B, E), tgt: (T, B, E), src_mask: (S, S), tgt_mask: (T, T), memory_mask: (T, S), src_key_padding_mask: (B, S), tgt_key_padding_mask: (B, T), memory_key_padding_mask: (B, S)
        '''
        device = x.device
        B = x.size(0)

        memory, memKeyPadMask, memMask = self.runEncoder(x) #memory shape is: Lenc x B x E

        predictions = []
        for b in range(B):
            if self.beamSize == 0:
                prediction = evaluate.evaluateXfmrSingleBatch(self, memory[:,b:b+1,:], memMask, memKeyPadMask[b:b+1,:])
            else:
                prediction = evaluate.evaluateXfmrSingleBatchBeamSearch(self, memory[:,b:b+1,:], memMask, memKeyPadMask[b:b+1,:], beamSize=self.beamSize)
            predictions.append(prediction)
        #pad predictions (inorder to convert into a tensor)
        # predictions, lens = nn.utils.rnn.pad_packed_sequence(predictions, batch_first=True, padding_value=decoder.pad_token) #BxL
        max_len = max([p.shape[1] for p in predictions])
        predictions_pad = torch.zeros(size=(B,max_len), dtype=torch.int32, device=device) #BxLdec
        for b in range(B):
            n = predictions[b].shape[1]
            predictions_pad[b:b+1,:n] = predictions[b]
        return predictions_pad #B x Ldec

