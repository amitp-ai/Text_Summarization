"""
    In an web browser, enter:
    https://0.0.0.0:8888/summarize?inputFileURL=https://public-text-summarizer.s3.amazonaws.com/inferenceData.json
    or using an EC2
    http://ec2-18-204-216-235.compute-1.amazonaws.com:8888/summarize?inputFileURL=https://public-text-summarizer.s3    .amazonaws.com/inferenceData.json
"""
from flask import Flask, request, jsonify
import requests
import time
import torch
import sys
sys.path.append('./src')
import utils
import inference
import models
import loadAndPreprocessData
import wandb
# wandb.login() #set 'export WANDB_API_KEY=key' environment variable and then run this script from that same shell


app = Flask(__name__)

@app.route("/")
def index():
    """Provide simple health check route"""
    return "This is a Text Summarization API...!"

@app.route("/summarize", methods=["GET"])
def summarize():
    """Summarize the input text"""
    global MODEL, LANG_TRAIN, WANDBRUN, LOGGER

    #Download and Preprocess the Input Data
    t1 = time.time()
    inputFileURL = request.args.get("inputFileURL")
    r = requests.get(inputFileURL, allow_redirects=True)
    inputTextFile = './Data/inferenceData.json'
    open(inputTextFile, 'wb').write(r.content)
    # inputTextFile = f"./Data/{inputTextFile}"
    descData, descVocabSize, absVocabSize, absIdx2Word, LOGGER = \
                loadAndPreprocessData.getData(inputTextFile=inputTextFile, 
                lang_train=LANG_TRAIN, logger=LOGGER)
    t2 = time.time()


    #Run Inference
    device = next(MODEL.parameters()).device
    LOGGER = inference.modelInference(model=MODEL, descData=descData, 
            abs_idx2word=absIdx2Word, device=device, logger=LOGGER)
    t3 = time.time()

    #Logging
    LOGGER['Data Loading Duration (s)'] = round(t2-t1, 3)
    LOGGER['Model Inference Duration (s)'] = round(t3-t2, 3)
    LOGGER['Time_Stamp'] = time.strftime("%H:%M:%S on %Y/%m/%d")
    LOGGER.toCSV('./Data/api.csv')
    #Also log into wandb as a summary
    WANDBRUN.summary.update(LOGGER.data) #will have to create a table in WandB to store after every run

    return jsonify({'Generated Summary is: ': LOGGER['Prediction_Summary'], 
                    'Rouge Score is': LOGGER['Rouge_Scores']})


def Pipeline(config):
    """Run the app"""
    global MODEL, LANG_TRAIN, WANDBRUN, LOGGER
    #Setup WandB
    t0 = time.time()
    WANDBRUN = wandb.init(project="Text-Summarization", notes="flask api", 
            tags=["app", "flask"], config=config, save_code=False, group='API', job_type=config['loadModelName'])
    config = WANDBRUN.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1 = time.time()

    #Load Vocabulary, Word2Idx, Idx2Word
    _, _, LANG_TRAIN = utils.loadWandBDataArtifact(artifactName='Data', 
        wandbRun=WANDBRUN, version='latest', trainData=False)
    descVocabSize = len(LANG_TRAIN.desc_vocab)
    absVocabSize = len(LANG_TRAIN.abs_vocab)
    t2 = time.time()

    #Load Model
    MODEL = eval('models.'+config.modelType)(descVocabSize=descVocabSize, absVocabSize=absVocabSize, 
                beamSize=config.beamSize, embMult=config.embMult, predMaxLen=config.predMaxLen, 
                encMaxLen=config.encMaxLen, pad_token=config.padToken, 
                hiddenDim=config.hiddenDim, numLayers=config.numLayers, dropout=config.dropout,
                numHeads=config.numHeads, decNumLayers=config.decNumLayers)
    MODEL, step, metricVal = utils.loadWandBModelArtifact(artifactNameVer=config.loadModelName, wandbRun=WANDBRUN, 
            model=MODEL, device=device, return_step=True)
    #convert model to static graph for faster inference
    # MODEL = torch.jit.script(MODEL) #doesn't work. gives error.
    LOGGER['Model_Info'] = f'Loaded {config.loadModelName} MODEL for {MODEL.__class__.__name__}, which is from step {step} and metric value is {metricVal:.3f}'
    t3 = time.time()

    #Logging
    LOGGER['Data-LangTrain Loading Duration (s)'] = round(t2-t1, 3)
    LOGGER['Model Loading Duration (s)'] = round(t3-t2, 3)

    #Run the app
    # app.run()
    app.run(host='0.0.0.0', port=8888, debug=False)

    #end wandbrun
    WANDBRUN.finish()

if __name__ == '__main__':
    #define global variables
    MODEL = LANG_TRAIN = LOGGER = WANDBRUN = None

    LOGGER = utils.CSVLogger()
    LOGGER['Time_Stamp'] = time.strftime("%H:%M:%S on %Y/%m/%d")
    config = utils.read_params('./config.yaml')['App']
    cfgParams = config['OtherParams']
    cfgModel = config['Models'][cfgParams['modelType']]
    config = cfgParams
    config.update(cfgModel)
    LOGGER['config'] = config

    Pipeline(config)
