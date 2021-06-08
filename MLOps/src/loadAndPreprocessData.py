'''
Load and preprocess data
'''
## Data Cleaning/Pre-Processing

"""
The data cleaning process is as follows:-
1. Lower-case all the words. 

2. Replace all numbers with a single token named 'NUMBER#'. 
Because there are so many different types of numbers, this way we can manage the complexity without completely discarding numbers.

3. Remove tokens that look like punctuations but aren;t really e.g. //, ./, )/, etc. 
But we will keep true punctuations (.,;:'\/?!+=-*&^%$#) inorder for the model to better "understand" the underlying text that we will be summarizing.

4. Remove acccent words and words with special unicode characters

5. Fix abbreviations that aren't properly tokenized e.g. 'i', '.', 'e', '.' => 'i.e.'

6. Remove words with long underscores for figures e.g. '_______________________'


### A note on word tokenization:
The raw data is already in lower case and has punctuations spaced out from words 
e.g. **'This is book.'** is like **'This is book .'** So string.split(' ') works pretty well for tokenization for the most part. 

From my experimentations, there a few instances where spacy and nltk's word tokenizer gives different results, 
it appears as if the raw data is pre-processed to be used with a str.split(' ') as the preferred means of word tokenzation. 
It seems to be giving the best results. 
So that is what we will use; and as a side benefit, it is significantly faster than spacy or nltk.
"""

import unicodedata
import re
import spacy
import utils

PARENT_DIR = './'
def getData(inputTextFile, cpc_codes, logger):
    '''
    Function to input and preprocess the input text (assuming one example at a time i.e. not in a batch)
    args:
        inputTextFile: json file containing the input text to be summarized. 
                        "Description" key is for description. 
                        An optional 'Target_Summary' key-value pair can also be provided.
        cpc_codes: string for cpc code e.g. 'd' or 'e' or even 'de'
        logger: to log different things in a CSV file
    '''
    desc_vocab = utils.load_json(file_name = f"{PARENT_DIR}Data/Training/Dicts/desc_vocab_final_{cpc_codes}_after_preprocess_text.json")
    abs_vocab = utils.load_json(file_name = f"{PARENT_DIR}Data/Training/Dicts/abs_vocab_final_{cpc_codes}_after_preprocess_text.json")
    desc_word2idx = utils.load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/desc_{cpc_codes}_word2idx.json')
    abs_idx2word = utils.load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/abs_{cpc_codes}_idx2word.json', ifIdx2Word=True)
    abs_word2idx = utils.load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/abs_{cpc_codes}_word2idx.json')
    # desc_idx2word = utils.load_json(file_name = f'{PARENT_DIR}Data/Training/Dicts/desc_{cpc_codes}_idx2word.json', ifIdx2Word=True)
    
    inputText = utils.load_json(f"{PARENT_DIR}Data/{inputTextFile}")
    desc = inputText['Description']
    logger['Desc_Orig'] = desc

    desc = Text_PreProcessing(desc)
    logger['Desc_AfterPreProcess'] = desc

    desc = desc + ['--stop--'] #add stop token
    desc = utils.create_numpy_array(len(desc), desc_word2idx)(desc)

    if 'Target_Summary' in inputText:
        tgtSmry = inputText['Target_Summary']
        logger['TgtSmry_Original'] = tgtSmry

        tgtSmry = Text_PreProcessing(tgtSmry)
        logger['TgtSmry_AfterPreProcess'] = tgtSmry

        tgtSmry = ['--start--'] + tgtSmry + ['--stop--'] #add stop token
        tgtSmry = utils.create_numpy_array(len(tgtSmry), abs_word2idx)(tgtSmry)
    else:
        tgtSmry = None
        logger['TgtSmry_Original'] = tgtSmry
        logger['TgtSmry_AfterPreProcess'] = tgtSmry


    Data = utils.InferenceDataset(desc, tgtSmry)
    # get_mini_df
    # Mini_Data_Language_Info

    return Data, len(desc_vocab), len(abs_vocab), abs_idx2word, logger


nlp = spacy.blank("en") #spacy is faster than nltk for word_tokenizing at least
REPLACEMENT_STRING = '--oov--'
def get_abbreviations(text_list):
    '''
    e.g. 'i',' ','.', 'e', ' ', '.' => 'i.e.'
    There are even numbers like 1 . 3 => 1.3 then convert it to #number#
    Wasn't 100% successful with Regex... Hence this Python function
    The raw data has abbreviations broken into separate tokens. This function will combine them properly.
    Input (text_list): list of string
    output (text_list): list of string
    '''
    alphabets = set(c for c in 'abcdefghijklmnopqrstuvwxyz') #make it a set for fast access
    i = 0
    while i+1 < len(text_list):
        k = 0
        while i+k+1 < len(text_list) and text_list[i+k] in alphabets and text_list[i+k+1] == '.':
            k += 2
        k -= 2
        if k > 0:
            text_list[i] = ''.join(text_list[i:i+k+2])
            text_list[i+1:i+k+2] = ['']*(k+1)
            i += (k+1)+1 #note this means i will really be (k+1)+2 as there is i+=1 below
        i += 1
    text_list = [t for t in text_list if t != '']
    return text_list


def remove_tokens_with_letters_and_numbers(text_list):
    """
    E.g. 'this1234'
    The below regex does this but it doesn't work 100% of the time and it is slower. Hence this Python function.
        #to get rid of tokens with numbers and letters in it
        myRegEx3 = re.compile(r'((\d+[a-z]+\d+)|([a-z]+\d+[a-z]+)|([a-z]+\d+)|(\d+[a-z]+))')
        text = myRegEx3.sub(REPLACEMENT_STRING, text)
    """
    for i in range(len(text_list)):
        t = text_list[i]
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            text_list[i] = REPLACEMENT_STRING
    return text_list


def Clean_Text_Post_Tokenization(text_list):
    text_list = remove_tokens_with_letters_and_numbers(text_list)
    text_list = get_abbreviations(text_list)
    return text_list


def Word_Tokenize(text):
    '''
    Descr: Tokenize the input text
    Input: text is raw string and nlp is spacy object for word tokenization
    Output: List of str
    '''
    #Directly tokenize into words. No need for sentence tokenization first.
    # text = nlp(text) #no need to use this (its also much slower)
    # text = [t.text for t in text]
    # text = text.split(' ') #use string split() as the raw data is setup for it vs spacy/nltk's word tokenizer (as discussed above)
    text = text.split(' ') #use string split() as the raw data is setup for it vs spacy/nltk's word tokenizer (as discussed above)
    return text


def remove_accented_chars(text):
    #remove characters like 'Sómě Áccěntěd těxt'
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def Clean_Text(text):
    #convert to raw string
    text = text.encode('unicode-escape').decode()

    # #get rid of parenthesis (no don't do this! Keep parenthesis)
    # myRegEx0 = re.compile(r'[\[\(] (\w+) [\]\)]')
    # text = myRegEx0.sub(r'\1', text)

    # address cases where there is no space between punctuations e.g. ').' => ')' '.'
    myRegEx0 = re.compile(r'([\.\,\:\;])?([\]\)\}\>])([\.\,\:\;])') 
    text = myRegEx0.sub(r'\1 \2 \3', text)

    #to get rid of tokens with multiples of punctuations e.g. '&#;' (**do this second**)
    myRegEx1 = re.compile(r'[\&\,\;\:\\\/\?\!\+\=\%\$\#\.\(\)\{\}\[\]\-\*\@\^\|\~]{2,}')
    text = myRegEx1.sub(REPLACEMENT_STRING, text)

    #to get rid of very long underscores e.g. 'Fig1____________________Results'
    myRegEx2 = re.compile(r'\S*_(_)+_\S*')
    text = myRegEx2.sub(REPLACEMENT_STRING, text)

    #to get rid of unicode characters e.g. 'u\1895'
    myRegEx3 = re.compile(r'\S*\\[a-z]\w+\S*')
    text = myRegEx3.sub(REPLACEMENT_STRING, text)

    #these are cases where --oov-- is with other characters without space and the regex to remove them in one shot is complex and slow. 
    #So this takes care of such issues e.g. (st--oov--wv)
    myRegEx4 = re.compile(r'\S*(--oov--)\S*')
    text = myRegEx4.sub(REPLACEMENT_STRING, text)

    
    #replace all numbers with a single token as well as floating point numbers (**only do this at the very end**)
    number_replacement = ' --#number#-- '
    myRegEx5 = re.compile(r'\d+ \. \d+')
    text = myRegEx5.sub(number_replacement, text)
    myRegEx6 = re.compile(r'\s+\d+\s+')
    text = myRegEx6.sub(number_replacement, text)

    return text


def Text_PreProcessing(text):
    '''
    Descr: Pipeline to clean/pre-process the input text
    Input: str
    Output: list of str
    '''
    text = text.lower()
    text = remove_accented_chars(text)
    text = Clean_Text(text)
    text = Word_Tokenize(text)
    text = Clean_Text_Post_Tokenization(text)
    return text
