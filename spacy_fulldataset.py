import numpy as np 
import pandas as pd
import os 
import evaluate
from datasets import load_dataset
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import pytextrank
#import gensim
from rouge import Rouge
import argparse
from distutils.util import strtobool
from heapq import nlargest


def parse_args():
    """ Basic function for parsing arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="textrank", help="Method of extractive summarization.")
    parser.add_argument("--download", type=lambda x: bool(strtobool(x)), default=False, help="Download spACy data file.")
    args = parser.parse_args()
    return args

def eval_textrank(example):
    """
        https://derwen.ai/docs/ptr/sample/
    """ 
    doc = nlp(example["article"])
    tr = doc._.textrank
    summary = ""
    for sent in tr.summary(limit_phrases=20, limit_sentences=2):
        summary += str(sent)
    example["prediction"] = summary
    return example

def eval_heapq(example):
    """
        https://www.activestate.com/blog/how-to-do-text-summarization-with-python/
    """

    doc = nlp(example["article"])
    tokens = [token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*0.2)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    
    example["prediction"] = summary
    return example


def _compute_metrics(example):
    """
    """
    rouge = Rouge()
    pred = example["prediction"]
    ref = example["highlights"]

    score = rouge.get_scores(pred, ref)
    return score[0]

def main():
    args = parse_args()
    method = args.method
    if args.download:
        spacy.cli.download("en_core_web_lg")
    else:
        global nlp 
        nlp = spacy.load('en_core_web_lg')
    if method == "textrank":
        nlp.add_pipe("textrank")
    
    # load our dataset from HuggingFace, using the testing
    # data
    cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = cnn_dailymail['test']

    # eval using one of the above methods
    if method == "textrank":
        train_dataset_preds = train_dataset.map(eval_textrank, num_proc=64)
    else:
        train_dataset_preds = train_dataset.map(eval_heapq, num_proc=64)

    # compute ROUGE scores
    results = train_dataset_preds.map(_compute_metrics, num_proc=64)
    results_flatten = results.flatten()
    rouge_results = {"rouge-1": np.mean(list(results_flatten["rouge-1.f"])),
                     "rouge-2": np.mean(list(results_flatten["rouge-2.f"])),
                     "rouge-l" : np.mean(list(results_flatten["rouge-l.f"]))}


    print(rouge_results)


if __name__=='__main__':
    main()