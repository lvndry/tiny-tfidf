from collections import Counter

from nltk.corpus import stopwords as nltkstopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk
import numpy as np
import pandas as pd

import os
import string
import copy
import pickle
import re
import math

ROOT_FOLDER = os.getcwd() + '/queries'
CORPUS_SIZE = 0
DF = {}
tf_idf = {}
corpus = []
processed_corpus = []

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def log(param):
    print(param)


def init_corpus():
    for filename in os.listdir(ROOT_FOLDER):
        full_path = os.path.join(ROOT_FOLDER+"/"+filename)
        corpus.append((full_path, filename))
    global CORPUS_SIZE
    CORPUS_SIZE = len(corpus)


def remove_stopwords(query):
    stopwords = nltkstopwords.words('english')
    words = word_tokenize(query)
    new_text = ""
    for w in words:
        if w not in stopwords and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_useless_chars(query):
    useless_chars = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    query = np.char.replace(query, "'", "")

    for symbol in useless_chars:
        query = np.char.replace(query, symbol, '')

    return str(query)


def stemming(query):
    stemmer = PorterStemmer()
    tokens = word_tokenize(query)
    stemmed_text = ""
    for w in tokens:
        stemmed_text = stemmed_text + " " + stemmer.stem(w)
    return stemmed_text


def preprocess(query: str):  # order matters
    query = query.lower()
    query = remove_stopwords(query)
    query = remove_useless_chars(query)
    query = stemming(query)

    return query


def extract_docs_terms(corpus):
    for path, title in corpus:
        file = open(path, 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()
        text = word_tokenize(preprocess(text))
        processed_corpus.append(text)


def words_document_frequency():
    for i in range(CORPUS_SIZE):
        vocab = processed_corpus[i]
        for w in vocab:
            try:
                DF[w] += 1
            except:
                DF[w] = 0


def get_word_df(word):
    df = 0
    try:
        df = DF[word]
    except:
        pass
    return df


def get_vocab_tfidf():
    for i in range(CORPUS_SIZE):
        story_words = processed_corpus[i]
        counter = Counter(story_words)
        words_count = len(story_words)

        for word in np.unique(story_words):
            tf = counter[word]/words_count
            df = get_word_df(word)
            idf = np.log((CORPUS_SIZE+1)/(df+1))

            tf_idf[(i, word)] = tf*idf


def search_most_relevant_response(query):
    print("Query:", query)

    tokens = word_tokenize(preprocess(query))

    query_weights = {}

    for tfidf in tf_idf:
        doc_index, word = tfidf
        if word in tokens:
            try:
                query_weights[doc_index] += tf_idf[tfidf]
            except:
                query_weights[doc_index] = tf_idf[tfidf]

    query_weights = sorted(query_weights.items(),
                           key=lambda x: x[1], reverse=True)

    results = []

    # Only select the 10 best queries
    for index, score in query_weights[:10]:
        path, _ = corpus[index]
        results.append((path, score))

    f = open(results[0][0], 'r')
    best_result = f.read().strip()
    f.close()
    print(best_result)
    print("\n\n")
    print("Best results:")
    print(results)


init_corpus()
extract_docs_terms(corpus)
words_document_frequency()
get_vocab_tfidf()

search_most_relevant_response("how works seo ?")
