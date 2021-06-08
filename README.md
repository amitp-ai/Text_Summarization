# Deep Learning Based Abstractive Text Summarization

## Data Collection
The `DataCollection` directory contains the script used to get various text summarization datasets such as BigPatents dataset, CNN/Daily mail, Arxiv/PubMed scientific papers, Gigaword dataset. For this project, the `BigPatents` dataset is use. For more details, please refer to the Readme.md file inside the DataCollection directory

## Data Wrangling
The `DataWrangling` directory contains all the code used to load the dataset, then preprocess it by using various regular expressions, then generate the vocabulary, word2idx, idx2word dictionaries for both the description as well as the summary. Furthermore, data visualization is performed to understand various aspects of the input description and summary. Please refer to `Readme.md` and `step5_data_wrangling.ipynb` located inside this directory for details. 

## Literature Survey
Notebook named `Literature_Survey.ipynb` located inside the `LiteratureSurvey` directory contains various methods used for text summarization, both extractive and abstractive methods. It discusses various unsupervised methods as well as supervised learning based deep learning methods.

## Model Experimentation/Building
The notebook named `ModelBuilding.ipynb` inside the `ModelBuilding` directory contains the many different deep learning based text summarization models I built.

## **MLOps**
**This is the main directory containing the end-to-end pipeline for data preprocessing, model training, data and model versioning, logging metrics, as well as inference.** Please refer to the `Readme.md` file located inside the `MLOps` directory for details.
