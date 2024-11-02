import numpy as np 
import pandas as pd
import os 
from rouge import Rouge


import spacy 
import pytextrank

from pprint import PrettyPrinter #print in a pretty way 
pp = PrettyPrinter()

def compute_f1_score(df, nlp, num=20):
    rouge = Rouge()
    scores = []
    ans = ""

    for i in range(20):
        doc = nlp(df.article[i])
        for j in doc._.textrank.summary(limit_phrases=10, limit_sentences=1):
            ans+=str(j)
            score = rouge.get_scores(ans, df.highlights[i])
            score = score[0]
            scores.append(score["rouge-l"]["f"])

    return np.mean(scores)

def summary_score(df,nlp,prin=False):
    
    ans = "" # collecting the summary from the generator
    doc = nlp(df.article[:20]) #apply the pipeline
    rouge = Rouge()
    # take our top 20 and compute
    
    for i in doc._.textrank.summary(limit_phrases=10, limit_sentences=1): #get the summary
        ans+=str(i)
        score = rouge.get_scores(ans, df.highlights[20])

    #mean_scores = rouge.get_scores(ans, df.highlights.to_list(), avg=True)
        
    #return mean_scores


def load_datasets(path_to_data_folder):
    df_train = pd.read_csv(path_to_data_folder+'train.csv')
    df_test = pd.read_csv(path_to_data_folder+'test.csv')
    df_validation = pd.read_csv(path_to_data_folder+'validation.csv')

    # concat all of the dfs together
    df = pd.concat([df_train, df_test, df_validation], ignore_index=True)

    del df_train, df_test, df_validation

    return df

def main():
    df = load_datasets('data/')

    # make sure to run this command first
    #spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textrank")
    print(compute_f1_score(df,nlp))
    


if __name__=='__main__':
    main()