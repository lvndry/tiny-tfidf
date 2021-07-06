from nltk.corpus import stopwords as nltkstopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import Counter

import nltk
import numpy as np

import os
import signal
import sys

#######


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

#######

ROOT_FOLDER = os.getcwd() + '/queries'
CORPUS_SIZE = 0
DF = {}
tf_idf = {}
corpus = []
processed_corpus = []

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def init_corpus():
    for filename in os.listdir(ROOT_FOLDER):
        full_path = os.path.join(ROOT_FOLDER+"/"+filename)
        corpus.append((full_path, filename))
    global CORPUS_SIZE
    CORPUS_SIZE = len(corpus)


def remove_single_letter_words(query):
    words = word_tokenize(query)
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_stopwords(query):
    stopwords = nltkstopwords.words('english')
    words = word_tokenize(query)
    new_text = ""
    for w in words:
        if w not in stopwords and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_useless_chars(query):
    useless_chars = "!\"#$%&()*+-./:;<=>?@[\]^_`'{|}~\n"

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
    query = remove_single_letter_words(query)

    return query


def extract_docs_terms(corpus):
    for path, title in corpus:
        doc_name = os.path.splitext(title)[0]
        doc_name = doc_name.split("_")
        doc_name = " ".join(doc_name)

        file = open(path, 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()

        text = word_tokenize(preprocess(text))
        text += word_tokenize(preprocess(doc_name))

        processed_corpus.append(text)


def words_document_frequency():
    for i in range(CORPUS_SIZE):
        vocab = processed_corpus[i]
        for word in vocab:
            try:
                DF[word] += 1
            except:
                DF[word] = 1


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
            # We add 1 to df to avoid division by 0
            idf = np.log((CORPUS_SIZE)/(df+1))
            tf_idf[(i, word)] = tf*idf


def search_most_relevant_response(query):
    query_tokens = word_tokenize(preprocess(query))

    query_scores = {}

    for tfidf in tf_idf:
        doc_index, word = tfidf
        if word in query_tokens:
            try:
                query_scores[doc_index] += tf_idf[tfidf]
            except:
                query_scores[doc_index] = tf_idf[tfidf]

    query_scores = sorted(query_scores.items(),
                          key=lambda x: x[1], reverse=True)

    results = []

    # Only select the 10 best queries
    for index, score in query_scores[:10]:
        path, _ = corpus[index]
        results.append((path, score))

    return results


init_corpus()
extract_docs_terms(corpus)
words_document_frequency()
get_vocab_tfidf()

while True:
    query = input("Search: ")

    results = search_most_relevant_response(query)

    print("\n\n")

    if len(results) > 0:
        f = open(results[0][0], 'r')
        best_result = f.read().strip()
        f.close()

        print("Best result:")
        print(best_result)
        print("\n\n")
        print("Top results:")
        print(results)
    else:
        print("No results found.")

    print("\n\n")
