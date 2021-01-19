## Three dataset explored
> aa
>> bb

### Text Summarization
    1. CNN/Daily Mail
        - Can get the dataset easily from [https://huggingface.co/datasets/cnn_dailymail][website]
        - For Tensorflow, can also easily get the dataset from <https://www.tensorflow.org/datasets/catalog/cnn_dailymail>
        - The source that both Huggingface and Tensorflow uses (as cited in their documentation) is:
        a. https://github.com/abisee/cnn-dailymail (This is the best place to start/use as has links/instructions to get processed and unprocessed data)
        b. https://cs.nyu.edu/~kcho/DMQA/ (original unprocessed dataset. Stories has the data for text summarization. Queestions is used for QA task)
           https://github.com/deepmind/rc-data (the script used to generate the data)
        c. https://github.com/becxer/cnn-dailymail/ (suggested by (a) for unprocessed data and how to process it -- it gets it's data from (b))
        d. https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail (suggested by (a) for processed data. Processed data in binary format for Tensorflow. It also has unprocessed data (probably copied from (b))
        
    2. Gigaword dataset
    
    3. Opinions dataset
    
### Question-Answering

    1. Google NQ

    2. RACE

    3. SQUAD 2
