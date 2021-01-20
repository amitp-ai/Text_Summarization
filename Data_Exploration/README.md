## Capstone Project on Text Summarization.
I explored the following different datasets for this task (this submission is for unit 3.5.2)

    1. CNN/Daily Mail
        - For Pytorch, can get the dataset easily from <https://huggingface.co/datasets/cnn_dailymail>
        - For Tensorflow, can also easily get the dataset from <https://www.tensorflow.org/datasets/catalog/cnn_dailymail>
        - But for the sake of experience, I decided to collect the data myself from the source that both Huggingface and Tensorflow uses (as cited in their documentation).
        - In particular, I collected data from (b) below
        a. https://github.com/abisee/cnn-dailymail (This is the best place to start/use as has links/instructions to get processed and unprocessed data)
        b. https://cs.nyu.edu/~kcho/DMQA/ (original unprocessed dataset. Stories has the data for text summarization. Queestions is used for QA task)
           https://github.com/deepmind/rc-data (the script used to generate the data)
        c. https://github.com/becxer/cnn-dailymail/ (suggested by (a) for unprocessed data and how to process it -- it gets it's data from (b))
        d. https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail (suggested by (a) for processed data. Processed data in binary format for Tensorflow. It also has unprocessed data (probably copied from (b))
        e. This actuallly has the CNN/DM dataset grouped by train/val/test sets <https://github.com/harvardnlp/sent-summary>. Maybe use this instead as its much better than (a/b)?
        
    2. Gigaword dataset
        - I collected data from (a) below    
        a. Get data from <https://github.com/harvardnlp/sent-summary> (Tensorflow data (b) links here too)
        b. For tensorflow from <https://www.tensorflow.org/datasets/catalog/gigaword>
    
    3. Opinions dataset (this is too small though, under 20MB)
    
    4. Reddit data
        a. https://www.tensorflow.org/datasets/catalog/reddit (this is very large ~18GB!)
    
### Summary
    The CNN dataset has 92579 examples and is 392.4MB in size.
    The CNN dataset md5 checksum is: 85ac23a1926a831e8f46a6b8eaf57263

    The Daily Mail dataset has 219506 examples and is 979.2MB in size.
    The Daily Mail dataset md5 checksum is: f9c5f565e8abe86c38bfa4ae8f96fd72

    The Gigaword dataset has 3993608 examples and is 939.4MB in size.
    The Gigaword dataset md5 checksum is: 064e658dc0d42e0f721d03287a7a9913
