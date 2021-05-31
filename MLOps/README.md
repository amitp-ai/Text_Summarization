# Overview

## Installing
Go to directory containing setup.py
pip install -e .
If want to create a wheel build and .tar that you can share with someone do,
python setup.py sdist bdist_wheel


## Unit Testing
Run all commands from `Production` directory
e.g. `python -m pytest -s ./test/`
			`python ./src/train.py`
			`python ./test/test_train.py`
##Training
python ./src/train.py --hiddenDim 128 --numLayers 2 --batchSize 24 --numEpochs 100 --lr 1e-3 --dropout 0.0 \
                      --savedModelDir './SavedModels/MODEL_7' --modelType 'Seq2SeqwithXfmrMemEfficient' \ 
                      --loadBestModel False --beamSize 0 --configPath './config.yaml'

##Inference
python ./src/inference.py --hiddenDim 128 --numLayers 2 --dropout 0.0 \--inputTextFile 'inferenceData.json' --modelType 'Seq2SeqwithXfmrMemEfficient' \--loadModelName 'MODEL7_step_20500.pth.tar' --beamSize 0 --configPath './config.yaml'

##OLD STUFF
!python ./src/train.py --hiddenDim 128 --numLayers 2 --batchSize 24 --numEpochs 100 --lr 1e-3 --dropout 0.0 \
                        --savedModelDir './saved_models/MODEL_7' --printEveryIters 500 --tbDescr 'MODEL_7' \
                        --modelType 'models.Seq2SeqwithXfmrMemEfficient' --loadBestModel False --toTrain True \
                        --fullVocab True --trainSize 5000 --valSize 16 --seed 0 --tfThresh 0.0 --beamSize 0