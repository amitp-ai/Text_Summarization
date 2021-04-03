"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Various utility functions to be used for training a text summarizer
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data


# Load data (string)
def load_data_string(split_type, cpc_codes, fname=None):
    input_path = os.path.join('/', 'content', 'drive', 'My Drive', 'Colab Notebooks', 'UCSDX_MLE_Bootcamp', 'Text_Summarization_UCSD', 'Step5_12-5-1_DataWrangling', 'bigPatentPreprocessedData')
    if not fname:
        file_names = os.listdir(os.path.join(input_path,split_type,cpc_code))
        for fname in file_names:
            file_name = os.path.join(input_path,split_type,cpc_code,file_name)
            if file_name.endswith('.gz'):
                df = pd.read_json(file_name, compression='gzip')
                yield df
    else:
        file_name = os.path.join(input_path,split_type,cpc_codes,fname)
        df = pd.read_json(file_name, compression='gzip')
        yield df



# Load data (numpy array)
def load_data_numpy(split_type, cpc_codes, fname=None):
    input_path = os.path.join('/', 'content', 'drive', 'My Drive', 'Colab Notebooks', 'UCSDX_MLE_Bootcamp', 'Text_Summarization_UCSD', 'Step5_12-5-1_DataWrangling', 'bigPatentPreprocessedData')
    if not fname:
        file_names = os.listdir(os.path.join(input_path,split_type,cpc_code))
        for fname in file_names:
            file_name = os.path.join(input_path,split_type,cpc_code,file_name)
            if file_name.endswith('.npz'):
                data_np = np.load(file_name, allow_pickle=True)
                yield data_np
    else:
        file_name = os.path.join(input_path,split_type,cpc_codes,fname)
        data_np = np.load(file_name, allow_pickle=True)
        yield data_np


class bigPatentDataset(data.Dataset):
    def __init__(self, data, shuffle=True):
        self.desc = torch.from_numpy(np.concatenate(data[:,0], axis=0)).to(dtype=torch.int32)
        self.abst = torch.from_numpy(np.concatenate(data[:,1], axis=0)).to(dtype=torch.int32)
        self.cpc = torch.from_numpy(np.stack(data[:,2], axis=0)).to(dtype=torch.int8)
        self.orig_abs = data[:,3]
        if shuffle: self.shuffle()
        print(f'Data shape is: {self.desc.shape}, {self.abst.shape}, {self.cpc.shape}')

    def __len__(self):
        return len(self.desc)
    
    def __getitem__(self, idx):
        return (self.desc[idx], self.abst[idx], self.cpc[idx], self.orig_abs[idx])

    def move_to(self, device):
        self.desc = self.desc.to(device=device)
        self.abst = self.abst.to(device=device)
        self.cpc = self.cpc.to(device=device)

    def shuffle(self, new_size=None):
        if new_size==None: new_size=len(self)
        with torch.no_grad():
            idxs = torch.randperm(len(self))[:new_size]
            self.desc = self.desc[idxs]
            self.abst = self.abst[idxs]
            self.cpc = self.cpc[idxs]
            self.orig_abs = self.orig_abs[idxs]

    def memory_size(self):
        variables = self.__dict__.keys()
        tot_mem = 0
        for v_name in variables:
            if v_name == 'orig_abs': continue
            v = self.__dict__[v_name]
            temp = v.element_size() * v.nelement()
            tot_mem += temp
        print(f'Total data size is: {tot_mem/1e6:.3f}MB')



class Mini_Data_Language_Info(object):
    def __init__(self, mini_df, desc_word2idx=None, abs_word2idx=None, desc_idx2word=None, abs_idx2word=None, 
                desc_vocab=None, abs_vocab=None):
        if desc_word2idx is None and abs_word2idx is None:
            # create vocab
            self.desc_vocab = self.create_vocab(mini_df.description)
            self.abs_vocab = self.create_vocab(mini_df.abstract)
            print(f'Description vocab size is {len(self.desc_vocab)} and Abstract vocab size is {len(self.abs_vocab)}')

            # create word2idx and idx2word
            self.desc_word2idx, self.desc_idx2word = self.create_word2idx_and_idx2word(self.desc_vocab)
            self.abs_word2idx, self.abs_idx2word = self.create_word2idx_and_idx2word(self.abs_vocab)
            print(f'Description word2idx dict size is {len(self.desc_word2idx)} and Abstract word2idx dict size is {len(self.abs_word2idx)}')
        else:
            self.desc_word2idx=desc_word2idx
            self.abs_word2idx=abs_word2idx
            self.desc_idx2word=desc_idx2word
            self.abs_idx2word=abs_idx2word
            self.desc_vocab=desc_vocab
            self.abs_vocab=abs_vocab

        # create numpy array
        max_desc = mini_df.description.apply(len).max()
        max_abs = mini_df.abstract.apply(len).max()
        print(f'max length (before adding stop token) in mini_df.description is {max_desc} and in mini_df.abstract (before adding start/stop tokens) is {max_abs}')
        max_abs, max_desc = 150, 4000

        mini_df['np_desc'] = mini_df.description.apply(lambda text: text + ['--stop--']) #need to add this for description
        mini_df['np_desc'] = mini_df.np_desc.apply(self.create_numpy_array(max_desc, self.desc_word2idx))
        # mini_df['np_abs'] = mini_df.abstract.apply(lambda text: ['--start--'] + text + ['--stop--']) #DON'T DO THIS!! The data already has this.
        mini_df['np_abs'] = mini_df.abstract.apply(self.create_numpy_array(max_abs, self.abs_word2idx))
        cpc_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'y':8}
        mini_df['np_cpc'] = mini_df.cpc_code.map(cpc_dict)
        self.mini_data = mini_df[['np_desc', 'np_abs', 'np_cpc', 'original_abstract']].to_numpy() #this is of type object and not numpy array of type int32/float32 (it's like a ragged numpy array)
        print(self.mini_data.shape)

    def create_vocab(self, ds):
        vocab = set()
        ds.apply(vocab.update)
        vocab.update(['--null--', '--oov--', '--#number#--', '--start--', '--stop--']) #add these as they may not be in the data
        return vocab

    def create_word2idx_and_idx2word(self, vocab):
        word2idx = {'--null--': 0, '--oov--': 1, '--#number#--': 2, '--start--': 3, '--stop--': 4}
        idx = len(word2idx)
        for w in vocab:
            if w not in word2idx:
                word2idx[w] = idx
                idx += 1
        idx2word = {i:w for w,i in word2idx.items()}
        return (word2idx, idx2word)

    def create_numpy_array(self, max_len, word2idx):
        def helper(text):
            np_array = np.zeros([1,max_len], dtype=np.int32)
            for i,w in enumerate(text):
                if w not in word2idx: w = '--oov--'
                np_array[0,i] = word2idx[w]
            return np_array
        return helper

######################################################################
