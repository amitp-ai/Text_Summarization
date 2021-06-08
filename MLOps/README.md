# MLOps Overview
In this repo we use an end-to-end pipeline (using the Weights-and-Biases MLOps framework) to train text summarization models, version control the dataset and the trained models, log various metrics of the model during training and inference, and use the best performing model for inference.  

Information about the runs, model and data versioning, etc can be found at:  
`https://wandb.ai/amitp-ai/Text-Summarization?workspace=user-amitp-ai`

Note: Run all commands from the `MLOps` directory

The dataset, saved models, logs, wandb repo, and wandb artifacts are stored in the AWS S3 bucket named `ucsdx-textsummarization` (arn:aws:s3:::ucsdx-textsummarization)  

The `config.yaml` file contains various configuration parameters used for training, inference, as well as unit testing.

## Installing
Go to MLOps directory (i.e. directory containing this Readme.md)
1. Install all the dependencies using:  
`pip install -r requirements.txt`
2. Install setup.py    
`pip install -e .`
If want to create a wheel build and .tar that you can share with someone do,
python setup.py sdist bdist_wheel

			
## Training
Below is an example of a command to train from the command line. We can continue training from a previous run using the `loadBestModel` argument. More details on the runs can be found at the above wandb link.   
`python ./src/train.py --hiddenDim 128 --numLayers 2 --decNumLayers 4 --numHeads 4 --batchSize 24 \
--numEpochs 100 --lr 1e-3 --dropout 0.0 --savedModelBaseName 'MODEL1' --modelType 'Seq2SeqwithXfmrMemEfficient' \
--loadBestModel 'True' --beamSize 0 --configPath './config.yaml'`


## Inference
Below is an example of a command to run inference from the command line using the latest training run of a model named `MODEL1`. Instead of the latest run, older runs can also be using using v0, v1 etc. Moreover, the best performing model in a given run is used.  
`python ./src/inference.py --hiddenDim 128 --numLayers 2 --decNumLayers 4 --numHeads 4 \
--dropout 0.0 --inputTextFile 'inferenceData.json' --modelType 'Seq2SeqwithXfmrMemEfficient' \
--loadModelName 'MODEL1:latest' --beamSize 0 --configPath './config.yaml'`  
Inference data and results are logged into the above wandb link as well as in a csv file.  


## Transfer Data and SavedModels to/from S3
1. Once the model is trained, the data and saved models can be uploaded to an AWS S3 bucket using:  
    `./s3Bucket.sh upload_and_delete_locally`  
2. Before training the model and running inference, download the latest data and saved models from AWS S3 using:             `./s3Bucket.sh download`


## Unit Testing
    `python -m pytest -s ./test/`  
