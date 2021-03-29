import os
import pandas as pd
import numpy as np
import psutil
import multiprocessing
# from tqdm import tqdm


# Load data (string)
def load_data_string(split_type, cpc_codes, fname=None):
    input_path = os.path.join('/', 'content', 'gdrive', 'My Drive', 'Colab Notebooks', 'UCSDX_MLE_Bootcamp', 'Text_Summarization_UCSD', 'Step5_12-5-1_DataWrangling', 'bigPatentPreprocessedData')
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
    input_path = os.path.join('/', 'content', 'gdrive', 'My Drive', 'Colab Notebooks', 'UCSDX_MLE_Bootcamp', 'Text_Summarization_UCSD', 'Step5_12-5-1_DataWrangling', 'bigPatentPreprocessedData')
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

######################################################################

'''
Processes raw data using multiprocessing module to speed things up
'''


def content_reader_parallel(df, operation):
    '''
    Note: can't use global variables with multiprocessing.
    It will give wrong results esp when want to update the global variable
    '''
    num_cpus = psutil.cpu_count(logical=True)
    n = len(df)//num_cpus
    data_chunks = []
    for i in range(num_cpus):
        strt = i*n
        stp = strt + n
        if i == num_cpus-1:
            stp = len(df)
        data_chunks.append(df[strt:stp])
    p = multiprocessing.Pool(processes=num_cpus)
    result = p.map(operation, data_chunks)
    p.close()
    p.join()
    return result


def content_reader_sequential(df, operation):
    return operation(df)

######################################################################


'''
Text Processing
'''
def word2sent_tokenizer(text):
    output = []
    temp = []
    sent_splits = {'.', '?', '!', ':'}
    for w in text:
        temp.append(w)
        if w in sent_splits:
            output.append(temp)
            temp = []
    return output


def sent2word_tokenizer(list_sents):
    temp_list = []
    for sent in list_sents:
        temp_list += sent
    return temp_list


def words2text(list_words):
    text = ' '.join(list_words)
    return text


def sents2text(list_sents):
    temp_list = []
    for sent in list_sents:
        temp_list += sent
    text = ' '.join(temp_list)
    return text

######################################################################
