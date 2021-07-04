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
SAMPLE_LENGTH = 0
DF = {}
tf_idf = {}
corpus = []
processed_corpus = []

nltk.download('stopwords')
nltk.download('punkt')


def init_corpus():
    for filename in os.listdir(ROOT_FOLDER):
        full_path = os.path.join(ROOT_FOLDER+"/"+filename)
        corpus.append((full_path, filename))
    global SAMPLE_LENGTH
    SAMPLE_LENGTH = len(corpus)


def remove_stopwords(story):
    stopwords = nltkstopwords.words('english')
    words = word_tokenize(story)
    new_text = ""
    for w in words:
        if w not in stopwords and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_useless_chars(story):
    useless_chars = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    story = np.char.replace(story, "'", "")

    for symbol in useless_chars:
        story = np.char.replace(story, symbol, '')

    return str(story)


def stemming(story):
    stemmer = PorterStemmer()
    tokens = word_tokenize(story)
    stemmed_text = ""
    for w in tokens:
        stemmed_text = stemmed_text + " " + stemmer.stem(w)
    return stemmed_text


def preprocess(story: str):
    story = story.lower()
    story = remove_stopwords(story)
    story = remove_useless_chars(story)
    story = stemming(story)

    return story


def words_document_frequency():
    for i in range(SAMPLE_LENGTH):
        vocab = processed_corpus[i]
        for w in vocab:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])


def get_word_df(word):
    df = 0
    try:
        df = DF[word]
    except:
        pass
    return df


def get_vocab_tfidf():
    for i in range(SAMPLE_LENGTH):
        story_words = processed_corpus[i]
        counter = Counter(story_words)
        words_count = len(story_words)

        for word in np.unique(story_words):
            tf = counter[word]/words_count
            df = get_word_df(word)
            idf = np.log((SAMPLE_LENGTH+1)/(df+1))

            tf_idf[(i, word)] = tf*idf


def extract_docs_terms(corpus):
    sample = corpus[:SAMPLE_LENGTH]
    for path, title in sample:
        file = open(path, 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()
        text = word_tokenize(preprocess(text))
        processed_corpus.append(text)


def search_most_relevant_story(query):
    tokens = word_tokenize(preprocess(query))

    print("Query:", query)
    print(tokens)

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

    for index, score in query_weights[:10]:
        path, _ = corpus[index]
        results.append((path, score))

    f = open(results[0][0], 'r')
    text = f.read().strip()
    f.close()
    print(text)


init_corpus()
extract_docs_terms(corpus)
words_document_frequency()
get_vocab_tfidf()
search_most_relevant_story("best subject")
