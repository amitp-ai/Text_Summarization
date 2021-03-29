import utils
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
# from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import sklearn.feature_extraction.text as sktext



def get_top_eigenvector(matrix):
    prob = np.exp(matrix)
    prob = prob/(np.sum(prob, axis=0, keepdims=True)+1e-3)
    w,v = np.linalg.eig(prob)
    v = v/(v.sum(axis=0)+1e-5)
    return v[:,np.argmax(w)]



def compute_bleu(row):
    #use corpus bleu instead of sentence bleu for comparing the abstracts as it different than averaging sentence bleu
    #actuallly can't use it b'cse the #of sentences in the target abstract is not equal to # of sentence in the predicted abstract
    #http://www.nltk.org/api/nltk.translate.html##nltk.translate.bleu_score.sentence_bleu

    # ref = utils.word2sent_tokenizer(row['original_abstract'].split(' '))
    # pred = utils.word2sent_tokenizer(row['pred_abstract'].split(' '))
    # bleu = corpus_bleu([ref], pred)
    ref = row['original_abstract'].split(' ') #need to split according to above documentation
    pred = row['pred_abstract'].split(' ')
    bleu4 = sentence_bleu([ref], pred, weights=(1/4, 1/4, 1/4, 1/4)) #ref is frst argument per above documentation
    return bleu4


rouge_evaluator = Rouge()
def compute_rouge(rouge_type):
    '''
    rouge_type (str): rouge-1, rouge-2, rouge-l
    '''
    def rouge_score(row):
        #https://pypi.org/project/rouge/ (pred is first argument)
        #https://kavita-ganesan.com/what-is-rouge-and-how-it-works-for-evaluation-of-summaries/#.YFF4RftKhrR
        ref = row['original_abstract']
        pred = row['pred_abstract']
        rouge = rouge_evaluator.get_scores(pred, ref)[0][rouge_type]['f'] #(pred is first argument)
        return rouge
    return rouge_score

def text_rank_parallel_results(result):
    df = pd.concat([r for r in result])
    return df


class Text_Summarizer(object):
    def __init__(self, word_embeddings=None, algo_type=None, use_sent_tokenization=True):
        self.algo_type = algo_type
        self.use_sent_tokenization = use_sent_tokenization
        self.word_embeddings = word_embeddings
        if self.algo_type is None: self.algo_type = self.text_rank_summary


    def predict(self, split_type='train', cpc_codes='de', fname='data1_str_json.gz', use_prl=True, data_size='full'):
        data = utils.load_data_string(split_type=split_type, cpc_codes=cpc_codes, fname=fname)
        for df in data:
            if data_size != 'full': df = df[0:data_size].copy()
            if not use_prl:
                df = utils.content_reader_sequential(df, self.text_summarizer_helper)
            else:
                df = text_rank_parallel_results(utils.content_reader_parallel(df, self.text_summarizer_helper))   
            print(df[['bleu-4', 'rouge_1', 'rouge_2', 'rouge_l']].mean(axis=0))
            if data_size != 'full': break
        print("\nHere's an example summary...")
        print(f"Original Abstract:\n{df.iloc[0]['original_abstract']}")
        print(f"Predicted Abstract:\n{df.iloc[0]['pred_abstract']}")


    def text_summarizer_helper(self, df):
        if self.use_sent_tokenization:
            ds_text = df.description.apply(utils.word2sent_tokenizer)
        else:
            ds_text = df.description.apply(utils.words2text) #df.original_description
        df['pred_abstract'] = ds_text.apply(self.algo_type)
        df['bleu-4'] = df.apply(compute_bleu, axis=1)
        df['rouge_1'] = df.apply(compute_rouge('rouge-1'), axis=1)
        df['rouge_2'] = df.apply(compute_rouge('rouge-2'), axis=1)
        df['rouge_l'] = df.apply(compute_rouge('rouge-l'), axis=1)
        return df


    def text_rank_summary(self, text):
        eps = 1e-3
        topn = 3
        text_embed = []
        
        if self.word_embeddings: #use glove features
            num_words_not_in_glove = 0
            num_tot_words = 0
            embed_dim = len(self.word_embeddings['the'])
            # #non vectorized version for sanity check
            # del_sim_mat = np.zeros((len(text), len(text)))
            # for i in range(len(text)):
            #     for j in range(len(text)):
            #         if i != j:
            #             senti = sum([word_embeddings.get(w, np.zeros((embed_dim,))) for w in text[i]])/(len(text[i])+eps)
            #             sentj = sum([word_embeddings.get(w, np.zeros((embed_dim,))) for w in text[j]])/(len(text[j])+eps)
            #             del_sim_mat[i][j] = cosine_similarity(senti.reshape(1,100), sentj.reshape(1,100))[0,0]

            for sent in text:
                num_tot_words += sum([1 for w in sent])
                num_words_not_in_glove += sum([w not in self.word_embeddings for w in sent])
                sent_embed = sum([self.word_embeddings.get(w, np.zeros((embed_dim,))) for w in sent])/(len(sent)+eps)
                # sent_embed = sum([np.zeros((embed_dim,)) for w in sent])/(len(sent)+1e-3)
                text_embed.append(sent_embed)
            # print(f'num_words_not_in_glove {num_words_not_in_glove}, num_tot_words {num_tot_words}')
            text_embed = np.stack(text_embed, axis=1).T #Num_sents X E
        
        else: #use TF-IDF
            res = []
            for sent in text:
                res.append(utils.words2text(sent))
            vectorizer = sktext.TfidfVectorizer()
            X = vectorizer.fit_transform(res)
            # print(X.shape, type(X), X.mean(1).shape)
            text_embed = X.toarray() #convert from sparse to dense numpy array (shape: num_sents x hidden_dim)

        sim_mat = cosine_similarity(text_embed, text_embed)
        sim_mat -= np.eye(sim_mat.shape[0])
        # get_top_eigenvector(sim_mat)
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        # print(f'No. of sents is {len(scores)} and the sentence ranking scores are:\n {sorted([(i,scores[i]) for i in range(len(text))], key=lambda x: x[1], reverse=True)}')

        summary = sorted([(scores[i], sent) for i,sent in enumerate(text)], key=lambda x: x[0], reverse=True)[:topn]
        summary = [sent for p,sent in summary]
        summary = utils.sents2text(summary)
        # summary = utils.sent2word_tokenizer(summary)
        # print(np.allclose(scores, get_top_eigenvector(sim_mat)))
        return summary


if __name__ == '__main__':
    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    text_rank.TextRank(word_embeddings=word_embeddings).text_rank_main()