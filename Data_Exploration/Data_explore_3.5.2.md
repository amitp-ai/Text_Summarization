## Three dataset explored for Text Summarization

    1. CNN/Daily Mail
        - Can get the dataset easily from <https://huggingface.co/datasets/cnn_dailymail>
        - For Tensorflow, can also easily get the dataset from <https://www.tensorflow.org/datasets/catalog/cnn_dailymail>
        - The source that both Huggingface and Tensorflow uses (as cited in their documentation) is:
        a. https://github.com/abisee/cnn-dailymail (This is the best place to start/use as has links/instructions to get processed and unprocessed data)
        b. https://cs.nyu.edu/~kcho/DMQA/ (original unprocessed dataset. Stories has the data for text summarization. Queestions is used for QA task)
           https://github.com/deepmind/rc-data (the script used to generate the data)
        c. https://github.com/becxer/cnn-dailymail/ (suggested by (a) for unprocessed data and how to process it -- it gets it's data from (b))
        d. https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail (suggested by (a) for processed data. Processed data in binary format for Tensorflow. It also has unprocessed data (probably copied from (b))
        e. This actuallly has the CNN/DM dataset grouped by train/val/test sets <https://github.com/harvardnlp/sent-summary>. Maybe use this instead?
        
    2. Gigaword dataset
        a. Get data from <https://github.com/harvardnlp/sent-summary> (Tensorflow data (b) links here too)
        b. For tensorflow from <https://www.tensorflow.org/datasets/catalog/gigaword>
    
    3. Opinions dataset (this is too small though, under 20MB)
    
    4. Reddit data
        a. https://www.tensorflow.org/datasets/catalog/reddit (this is very large ~18GB!)
    
### Summary
    The CNN dataset has 92579 examples and is 392.367704MB in size.
    The Daily Mail dataset has 219506 examples and is 979.205688MB in size.
    The Gigaword dataset has 3993608 examples and is 939.402629MB in size.
