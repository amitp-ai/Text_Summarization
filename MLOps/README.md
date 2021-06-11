# MLOps Overview
In this repo we use an end-to-end pipeline (using the Weights-and-Biases MLOps framework) to train text summarization models, version control the dataset and the trained models, log various metrics of the model during training and inference, and use the best performing model for inference.  

Information about the runs, model and data versioning, etc can be found at:  
`https://wandb.ai/amitp-ai/Text-Summarization?workspace=user-amitp-ai`

Note: Run all commands from the `MLOps` directory

The Dataset, SavedModels, and logs directories are merely provided as an example. The full version-controlled datasets, saved models, logs, wandb repo, and wandb artifacts are stored in the AWS S3 bucket named `ucsdx-textsummarization` (arn:aws:s3:::ucsdx-textsummarization)  as well as in the above linked WandB project.

The `config.yaml` file contains various configuration parameters used for training, inference, as well as unit testing.

## Installing
Go to MLOps directory (i.e. directory containing this Readme.md)
1. Install all the dependencies using:  
`pip install -r requirements.txt`
2. Install setup.py    
`pip install -e .`
If want to create a wheel build and .tar that you can share with someone do,
python setup.py sdist bdist_wheel


## Transfer Data and SavedModels to/from S3 Bucket
First configure the AWS S3 bucket access by running `aws configure` and enter the `key id` and `access key` for accessing the bucket.
1. Once the model is trained, the data and saved models can be uploaded to an AWS S3 bucket using:  
    `./s3Bucket.sh upload_and_delete_locally`  
2. Before training the model and running inference, download the latest data and saved models from AWS S3 using:             `./s3Bucket.sh download`


## Unit Testing
    `python -m pytest -s ./test/`  

			
## Training
Below is an example command to run the training pipeline (i.e. data loading, data versioning, model training, logging metrics/artifacts, model versioing, etc) from the command line. We can continue training from a previous run using the `loadBestModel` argument. More details on the runs can be found at the above wandb link.  

`python ./src/train.py --hiddenDim 128 --numLayers 2 --decNumLayers 4 --numHeads 4 --batchSize 24 \
--numEpochs 100 --lr 1e-3 --dropout 0.0 --savedModelBaseName 'MODEL1' --modelType 'Seq2SeqwithXfmrMemEfficient' \
--loadBestModel 'True' --beamSize 0 --configPath './config.yaml'`


## Inference
Below is an example command to run inference pipeline from the command line using the latest training run of a model named `MODEL1`. Instead of the latest run, older runs can also be using using v0, v1 etc. Moreover, the best performing model in a given run is used.  

This is especially convenient because we can train/build various different models (using the above command), and once we are satisfied with a given model, we can use that model name in the below command to download it from WandB server and then deploy it into production.  

`python ./src/inference.py --hiddenDim 128 --numLayers 2 --decNumLayers 4 --numHeads 4 \
--dropout 0.0 --inputTextFile 'inferenceData.json' --modelType 'Seq2SeqwithXfmrMemEfficient' \
--loadModelName 'MODEL1:latest' --beamSize 0 --configPath './config.yaml'`  
Inference data and results are logged into the above wandb link as well as in a csv file.  


## Flask API
The `app` directory contains `app.py` which is a flask based API for model serving. Below is an example command to consume this API:  
`https://0.0.0.0:5000/summarize?inputFileURL=https://public-text-summarizer.s3.amazonaws.com/inferenceData.json`  
The json file containing the input text can be stored at any public repository. The format of this json file should be:   `{"Description": "Description text", "Target_Summary": "Target summary text"}.` `The Target_Summary` is an optional field that can be used for further model training.


## Production Deployment
For production deployment, use the Dockerfile in this directory to build a container using the following command:  
`docker build -t textsum/inference -f MLOps/Dockerfile .`  
For example, this Docker container can be deployed to an AWS EC2 with host IP 0.0.0.0 and port 5000 set to open. To do this, create new security group (named `full-access`) and set `inbound rule` to `all traffic` (this will also set host ip to 0.0.0.0/5000). Then select your instance, right click, select security, change security group and select the new group created (named `full access`). The model can then be consumed using the above Flask API example. 

If the WandB access key is already provided, then one can directly launch the container with the following command:  
`docker run -p 5000:5000 -it --rm --entrypoint bash summarize/api`  

Otherwise, launch the Docker container into a Bash shell using the following command:  
`docker run -p 5000:5000 -it --rm summarize/api`  
Then set the wandb key as an environment variable as follows:  
`export WANDB_API_KEY=<key>`  
Then launch the application by running:  
`python ./app/app.py`
