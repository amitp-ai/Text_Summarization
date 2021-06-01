"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
Description:
    Various utility functions and classes used for training a text summarizer
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import json
import logging
import queue
import shutil
import torch.autograd.profiler as profiler
import yaml

PARENT_DIR = './'

def create_logger(fileName, logDir=None, stdOut=True):
    '''
    create logger and add to it a file handler and a stream handler
    This will be helpful during production
    (if calling from Jupyter notebook and seeing unexpected logging behavior, restart the kernel)
    '''
    if not logDir:
        # logDir = os.path.join('/', 'content', 'drive', 'My Drive', 'Colab Notebooks', 'UCSDX_MLE_Bootcamp', 'Text_Summarization_UCSD', 'ModelBuilding', 'logs')
        logDir = PARENT_DIR + 'logs'
    #create logger and log everything (debug and above)
    logger = logging.getLogger(fileName.strip('.log'))
    logger.setLevel(logging.DEBUG)
    logPath = os.path.join(logDir, fileName)

    #create filehandler to save to .log file and also create format for the logs 
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s', datefmt='%m.%d.%y %H:%M:%S')
    fileHandler = logging.FileHandler(logPath)
    fileHandler.setLevel(logging.DEBUG) #not necessary as usually used to set a different level from one set above
    fileHandler.setFormatter(formatter)
    #add filehandles and streamhandler to the logger
    logger.addHandler(fileHandler)

    if stdOut:
        #create streamhandler to also print to the console and also create format for the logs 
        formatter = logging.Formatter('%(message)s')    
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG) #not necessary as usually used to set a different level from one set above
        streamHandler.setFormatter(formatter)
        #add filehandles and streamhandler to the logger
        logger.addHandler(streamHandler)

    return logger

def closeLoggerFileHandler(logger):
    ''' To release file handler when done. 
    Esp need this when used from Jupyter Notebook '''
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def load_data_string(split_type, cpc_codes, fname=None):
    """
    Load data (in string form)
    """
    input_path = PARENT_DIR + 'Data/Training/Preprocessed/'
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



def load_data_numpy(split_type, cpc_codes, fname=None):
    """
    Load data (as a numpy array)
    """
    input_path = PARENT_DIR + 'Data/Training/Preprocessed/'
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


def get_mini_df(data, mini_df_size=16, verbose=False):
    '''
    Get a small subset of data for quick eval/debug
    '''
    for df in data:
        if verbose:
            print(df.head(5), df.shape, df.columns)
            print(df.iloc[0,0])
            print(df.iloc[0,1])
        mini_df = df.iloc[:mini_df_size,:].copy() #create a small dataset for fast prototyping and ease of debugging
        return mini_df


def load_json(file_name, ifIdx2Word=False):
    """
    Load json file
    """
    with open(file_name, "r") as fh:
        jsonDict = json.load(fh)
    if ifIdx2Word:
        jsonDict = {int(idx):w for idx,w in jsonDict.items()} #have to do this but not sure why don't have to do this for word2idx (need to check how json load works)
    return jsonDict



class bigPatentDataset(data.Dataset):
    def __init__(self, data, shuffle=True):
        self.desc = torch.from_numpy(np.concatenate(data[:,0], axis=0)).to(dtype=torch.int32)
        self.abst = torch.from_numpy(np.concatenate(data[:,1], axis=0)).to(dtype=torch.int32)
        self.cpc = torch.from_numpy(np.stack(data[:,2], axis=0)).to(dtype=torch.int8)
        self.orig_abs = data[:,3]
        self.orig_desc = data[:,4]
        if shuffle: self.shuffle()
        print(f'Data shape is: {self.desc.shape}, {self.abst.shape}, {self.cpc.shape}')

    def __len__(self):
        return len(self.desc)
    
    def __getitem__(self, idx):
        return (self.desc[idx], self.abst[idx], self.cpc[idx], self.orig_abs[idx], 0) #self.orig_desc[idx])

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
            self.orig_desc = self.orig_desc[idxs]

    def memory_size(self):
        variables = self.__dict__.keys()
        tot_mem = 0
        for v_name in variables:
            if 'orig_' in v_name: continue
            v = self.__dict__[v_name]
            temp = v.element_size() * v.nelement()
            tot_mem += temp
        print(f'Total data size is: {tot_mem/1e6:.3f}MB')


class InferenceDataset(data.Dataset):
    def __init__(self, dataDesc, dataTgtSmry):
        self.desc = torch.from_numpy(dataDesc).to(dtype=torch.int32)
        self.tgtSmry = torch.from_numpy(dataTgtSmry).to(dtype=torch.int32) if dataTgtSmry is not None else None

    def __len__(self):
        return len(self.desc)
    
    def __getitem__(self, idx):
        return (self.desc[idx], self.tgtSmry[idx] if self.tgtSmry is not None else None)

    def move_to(self, device):
        self.desc = self.desc.to(device=device)
        if self.tgtSmry is not None: self.tgtSmry.to(device=device)

class Mini_Data_Language_Info(object):
    """
    Only use this if not using the full dataset.
    Otherwise the vocab for the full dataset is already available and stored as part of datawrangling process
    """
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
        mini_df['np_desc'] = mini_df.np_desc.apply(create_numpy_array(max_desc, self.desc_word2idx))
        # mini_df['np_abs'] = mini_df.abstract.apply(lambda text: ['--start--'] + text + ['--stop--']) #DON'T DO THIS!! The data already has this.
        mini_df['np_abs'] = mini_df.abstract.apply(create_numpy_array(max_abs, self.abs_word2idx))
        cpc_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'y':8}
        mini_df['np_cpc'] = mini_df.cpc_code.map(cpc_dict)
        self.mini_data = mini_df[['np_desc', 'np_abs', 'np_cpc', 'original_abstract', 'original_description']].to_numpy() #this is of type object and not numpy array of type int32/float32 (it's like a ragged numpy array)
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

def create_numpy_array(max_len, word2idx):
    def helper(text):
        np_array = np.zeros([1,max_len], dtype=np.int32)
        for i,w in enumerate(text):
            if w not in word2idx: w = '--oov--'
            np_array[0,i] = word2idx[w]
        return np_array
    return helper

######################################################################

### Saving and Loading Models ###

class CheckpointSaver(object):
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `maxCheckpoints` have been saved.

    Args:
        saveDir (str): Directory to save checkpoints.
        metricName (str): Name of metric used to determine best model.
        maximizeMetric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        logger (logging.Logger): Optional logger for printing information.
        maxCheckpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
    >> adopted from CS224n Squad2.0 project -- util.py
    """
    def __init__(self, savedModelBaseName, metricName, maximizeMetric, logger, maxCheckpoints=3, bestVal=None):
        super(CheckpointSaver, self).__init__()

        self.baseModelName = f"./SavedModels/{savedModelBaseName}"
        self.maxCheckpoints = maxCheckpoints
        self.metricName = metricName
        self.maximizeMetric = maximizeMetric
        self.logger = logger
        self.bestVal = bestVal
        self.ckptPaths = queue.PriorityQueue()
        self.logger.debug(f"Saver will {'max' if maximizeMetric else 'min'}imize {metricName}...")

    def isBest(self, metricVal):
        """Check whether `metricVal` is the best seen so far.

        Args:
            metricVal (float): Metric value to compare to prior checkpoints.
        """
        if metricVal is None:
            # No metric reported
            return False

        if self.bestVal is None:
            # No checkpoint saved yet
            return True

        return ((self.maximizeMetric and self.bestVal < metricVal)
                or (not self.maximizeMetric and self.bestVal > metricVal))

    def save(self, step, model, metricVal, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metricVal (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckptDict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step,
            'metric_val': metricVal
        }
        model.to(device) #move back to the device the model resided before moving it to cpu for ckptDict

        checkpointPath = f'{self.baseModelName}_step_{step}.pth.tar'
        torch.save(ckptDict, checkpointPath)
        self.logger.debug(f'Saved checkpoint: {checkpointPath}')

        if self.isBest(metricVal):
            # Save the best model
            self.bestVal = metricVal
            best_path = f'{self.baseModelName}_best.pth.tar'
            shutil.copy(checkpointPath, best_path)
            self.logger.debug(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximizeMetric: #otherwise minimize metric
            priority_order = metricVal
        else:
            priority_order = -metricVal

        self.ckptPaths.put((priority_order, checkpointPath))

        # Remove a checkpoint if more than maxCheckpoints have been saved
        if self.ckptPaths.qsize() > self.maxCheckpoints:
            _, worst_ckpt = self.ckptPaths.get()
            try:
                os.remove(worst_ckpt)
                self.logger.debug(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                self.logger.debug(f'Unable to remove checkpoint: {worst_ckpt}... maybe its already removed or protected')


def loadModel(model, checkpointPath, device, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.Module): Load parameters into this model.
        checkpointPath (str): Path to checkpoint to load.
        Device (str): cuda or cpu
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.Module): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    ckptDict = torch.load(f"./SavedModels/{checkpointPath}", map_location=device)

    # load parameters into the model
    model.load_state_dict(ckptDict['model_state'])

    if return_step:
        step = ckptDict['step']
        metricVal = ckptDict['metric_val']
        return model, step, metricVal

    return model

''' ######################################### '''

#!pip install nvidia-ml-py3
# import nvidia_smi
def GPU_Memory_Usage():
    '''
    # print('nvidia-smi:')
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'memory used: {mem_res.used / (1024**3)} (GiB) i.e. {100 * (mem_res.used / mem_res.total):.3f}%') # usage in GiB/percentage
    '''

    print('Pytorch memory info:')
    print(f'memory_allocated: {torch.cuda.memory_allocated(device=None)/1e6}MB')
    print(f'memory_cached: {torch.cuda.memory_cached(device=None)/1e6}MB')


def getModelInfo(model):
    for n,m in model.named_children():
        num_params = 0
        for p in m.parameters():
            if p.requires_grad:
                num_params += p.numel()
        print(f'{n}: {num_params} parameters')


def profileModel(model, data, devName='cuda'):
    '''to profile a Pytorch model'''
    #data prep
    device = torch.device(devName)
    batch = 6
    data = data[:batch]
    x,y = data[0].to(device), data[1].to(device)
    model = model.to(device=device)
    useCuda = True if devName=='cuda' else False


    #compute profile
    with profiler.profile(use_cuda=useCuda, record_shapes=True) as prof:
        with profiler.record_function("Model Forward Pass"):
            model(x, y)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=f"{devName}_time_total", row_limit=10, top_level_events_only=False))

    #memory profile
    with profiler.profile(use_cuda=useCuda, profile_memory=True, record_shapes=True) as prof:
        model(x, y) 
    print(prof.key_averages().table(sort_by=f"{devName}_memory_usage", row_limit=10, top_level_events_only=False))

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(use_full_vocab, cpc_codes, fname, train_size, val_size, logger):
    """Get data for training"""
    logger.debug('Getting the training and validation data...')
    #for desc and abs: get the vocab, word2idx, idx2word
    desc_vocab = load_json(file_name = f"{PARENT_DIR}Data/Training/Dicts/desc_vocab_final_{cpc_codes}_after_preprocess_text.json")
    abs_vocab = load_json(file_name = f"{PARENT_DIR}Data/Training/Dicts/abs_vocab_final_{cpc_codes}_after_preprocess_text.json")

    desc_word2idx = load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/desc_{cpc_codes}_word2idx.json')
    abs_word2idx = load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/abs_{cpc_codes}_word2idx.json')

    desc_idx2word = load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/desc_{cpc_codes}_idx2word.json', ifIdx2Word=True)
    abs_idx2word = load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/abs_{cpc_codes}_idx2word.json', ifIdx2Word=True)

    #get the training and val data
    data_train = load_data_string(split_type='train', cpc_codes=cpc_codes, fname=fname)
    data_val = load_data_string(split_type='val', cpc_codes=cpc_codes, fname=fname)
    mini_df_train = get_mini_df(data_train, mini_df_size=train_size) 
    mini_df_val = get_mini_df(data_val, mini_df_size=val_size) 

    if use_full_vocab:
        lang_train = Mini_Data_Language_Info(mini_df_train, desc_word2idx=desc_word2idx,abs_word2idx=abs_word2idx,
                                                desc_idx2word=desc_idx2word, abs_idx2word=abs_idx2word,
                                                desc_vocab=desc_vocab, abs_vocab=abs_vocab) #using full vocab
    else:
        lang_train = Mini_Data_Language_Info(mini_df_train) #generate vocab etc (i.e. don't use full vocab)
    logger.debug(f'Size of description vocab is {len(lang_train.desc_vocab)} and abstract vocab is {len(lang_train.abs_vocab)}')
    lang_val = Mini_Data_Language_Info(mini_df_val, desc_word2idx=lang_train.desc_word2idx,abs_word2idx=lang_train.abs_word2idx)

    train_data = bigPatentDataset(lang_train.mini_data, shuffle=True)
    train_data.memory_size()
    val_data = bigPatentDataset(lang_val.mini_data, shuffle=True)

    return train_data, val_data, lang_train