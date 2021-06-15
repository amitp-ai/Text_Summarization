"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Contains various models for text summarization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src import utils, evaluate

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
        div_term = hpar ** (2*torch.arange(0,dModel,2).true_divide(dModel)).to(dtype=torch.float32).unsqueeze(0) #1 x true_divide2

        pe[:, 0::2] = torch.sin(position.true_divide(div_term)) #maxLen x d/2
        pe[:, 1::2] = torch.cos(position.true_divide(div_term)) #maxLen x d/2
        pe = pe.unsqueeze(0)  #1 x maxLen x d
        self.register_buffer('pe', pe) #not trainable but saved with the model and moved to cpu/gpu with model.to()

    def forward(self, x):
        '''x: B x Len'''
        x = torch.zeros_like(x).unsqueeze(-1) + self.pe[:, :x.size(1), :] #B x len x d
        return x


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
    def __init__(self, descVocabSize, absVocabSize, hiddenDim, beamSize, embMult, 
                numHeads, dropout, numLayers, decNumLayers, predMaxLen, encMaxLen, pad_token):
        super(Seq2SeqwithXfmrMemEfficient, self).__init__()
        self.hidDim = hiddenDim
        self.embDim = hiddenDim
        self.numHeads = numHeads
        self.beamSize = beamSize
        self.embMult = embMult #hiddenDim (hiddenDim seems to work worse than 4)
        self.dropout = dropout
        self.numLayers = numLayers
        self.decNumLayers = decNumLayers
        self.predMaxLen = predMaxLen
        self.encMaxLen = encMaxLen
        self.pad_token = pad_token
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

    def runEncoderHelper(self, xsub):
        xMask = (xsub != self.pad_token).unsqueeze(-1) #B x Lenc/n x 1
        xsub = self.encEmbedding(xsub)*np.sqrt(self.embMult) + self.encPos(xsub) #B x Lenc/n x E
        xsub = xsub * xMask #zero out all the inputs corresponding to pad tokens
        xsub= xsub.transpose(0,1) #Lenc/n x B x E

        srcKeyPadMask = xMask.logical_not().squeeze(-1) #B x Lenc/n
        srcKeyPadMask[:,0] = False #assume first position is non pad (otherwise self.encoder() retuns tensor with nan)
        srcMask = None #bcse causality does not apply (so attend to everything)
        memory = self.encoder(xsub, mask=srcMask, src_key_padding_mask=srcKeyPadMask) #Lenc/n x B x E

        return memory, srcKeyPadMask

    def runEncoder(self, x):
        ''' x is of shape : B x Lenc'''
        numSplits, overLap = 4, 100
        Lenc = x.size(1)
        split = Lenc//numSplits
        memory = []
        memKeyPadMask = []
        for i in range(numSplits):
            strt, stp = i*split, (i+1)*split + overLap
            xTmp = x[:,strt:stp] #B x Lenc/n
            memoryTmp, srcKeyPadMaskTmp = self.runEncoderHelper(xTmp)
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

