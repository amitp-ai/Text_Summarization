# Deep Learning Based Abstractive Text Summarization
This project is about building a text summarization system (for legal documents) where the objective is to read in a piece of text (potentially containing many paragraphs) and output a summarized version of it. A good summarizer will output all the important details from the input text while being succinct.

There are many places where a good document summarizer will be valuable. For example, in the legal industry it can be used to summarize long legal documents, in the healthcare industry it can be used to summarize important aspects of a medication, in the news industry it can be used to summarize news articles, in the financial industry it can be used to summarize 10K SEC filings, and many other examples.

In terms of approaches, there are two different types to text summarization approaches:
1. Extractive text summarization: It identifies important sections of the original article and then copies it to form the summary. *It can be thought of as a highlighter.*
2. Abstractive text summarization: It reproduces important information in the article by first understanding the entire article and then succinctly generating new text based upon it. *It can be thought of as a pen.*

Out of the two, abstractive summarization is more like what humans do; and thus, it has greater potential. But the downside is that it is much more complicated to implement because it requires a language model to generate new text based upon some prior (i.e. the input article). For this project, an abstractive text summarizer (using deep learning) is developed.

The outline of the project is as follows:
1. Data collection
2. Exploratory Data Analysis and Data Wrangling
3. Literature Survey
4. Model Building
5. MLOps and Deployment


## Data Collection
The `DataCollection` directory contains the script used to get various text summarization datasets such as BigPatents dataset, CNN/Daily mail, Arxiv/PubMed scientific papers, Gigaword dataset. For this project, the `BigPatents` dataset is use. For more details, please refer to the Readme.md file inside the DataCollection directory

## Exploratory Data Analysis and Data Wrangling
The `DataWrangling` directory contains all the code used to load the dataset, then preprocess it by using various regular expressions, then generate the vocabulary, word2idx, idx2word dictionaries for both the description as well as the summary. Furthermore, data visualization is performed to understand various aspects of the input description and summary. Please refer to `Readme.md` and `step5_data_wrangling.ipynb` located inside this directory for details. 

## Literature Survey
Notebook named `Literature_Survey.ipynb` located inside the `LiteratureSurvey` directory contains various methods used for text summarization, both extractive and abstractive methods. It discusses various unsupervised methods such as TextRank, Lead-3, Random Sampling as well as supervised learning based deep learning methods such as Pointer-Generator Networks, pre-trained BERT based models, etc.

## Model Building/Experimentation
The notebook named `ModelBuilding.ipynb` inside the `ModelBuilding` directory contains the many different encoder-decoder types of deep learning based text summarization models I built, such as:
- LSTM
- LSTM with Attention
- Transformers
- Memory efficient transformers

## MLOps
This is the main directory containing the end-to-end pipeline for data preprocessing, model training, data & model versioning, logging metrics, as well as inference and production deployment. The `Weights and Biases` framework is used for MLOps and `Pytorch` is used for model training. Flask based API is used for model serving. Moreover, a Docker container is built that can then be deployed to a production environment. Please refer to the `Readme.md` file located inside the `MLOps` directory for details.
