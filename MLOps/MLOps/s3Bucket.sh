#!/bin/bash

# ***Note: Be careful as upload/download will completely replace target files with source files.

echo "s3 bucket name: s3://ucsdx-textsummarization"
aws s3 ls s3://ucsdx-textsummarization --human-readable --summarize
if [ "$1" == "download" ]; then
    echo "downloading data, models, artifacts, wandb, & logs directories from s3..."
    rm -rf ./Data ./SavedModels ./artifacts ./wandb ./logs
    aws s3 cp s3://ucsdx-textsummarization/Data ./Data --recursive
    aws s3 cp s3://ucsdx-textsummarization/SavedModels ./SavedModels --recursive
    aws s3 cp s3://ucsdx-textsummarization/artifacts ./artifacts --recursive
    aws s3 cp s3://ucsdx-textsummarization/wandb ./wandb --recursive
    aws s3 cp s3://ucsdx-textsummarization/logs ./logs --recursive
    # aws s3 sync s3://ucsdx-textsummarization/Data ./Data --delete # sync does not work well
elif [ "$1" == "upload_and_delete_locally" ]; then
    echo "uploading data, models, artifacts, wandb, & logs directories to s3 and then deleting them from the local directory..."
    aws s3 rm s3://ucsdx-textsummarization/ --recursive
    # aws s3 rm s3://ucsdx-textsummarization/Data --recursive #no need to do it separately
    aws s3 cp ./Data s3://ucsdx-textsummarization/Data --recursive
    aws s3 cp ./SavedModels s3://ucsdx-textsummarization/SavedModels --recursive
    aws s3 cp ./artifacts s3://ucsdx-textsummarization/artifacts --recursive
    aws s3 cp ./wandb s3://ucsdx-textsummarization/wandb --recursive
    aws s3 cp ./logs s3://ucsdx-textsummarization/logs --recursive
    rm -rf ./Data ./SavedModels ./artifacts ./wandb ./logs
    # aws s3 sync ./SavedModels/ s3://ucsdx-textsummarization/SavedModels --delete # sync does not work well
elif [ "$1" == "ls" ]; then
    echo "recursively listing files and directories in ucsdx-textsummarization s3 bucket..."
    aws s3 ls s3://ucsdx-textsummarization --human-readable --summarize --recursive
    aws s3 ls s3://ucsdx-textsummarization --human-readable --summarize
elif [ "$1" == "rm" ]; then
    echo "deleting everything from ucsdx-textsummarization s3 bucket..."
    aws s3 rm s3://ucsdx-textsummarization/ --recursive
else
    echo "invalid parameter"
fi
