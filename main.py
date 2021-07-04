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

ROOT_FOLDER = os.getcwd() + '/stories'
corpus = []

nltk.download('stopwords')
nltk.download('punkt')


def init_corpus():
    file = open(ROOT_FOLDER+"/index.html", 'r')
    text = file.read().strip()
    file.close()

    file_name = re.findall('><A HREF="(.*)">', text)
    file_title = re.findall('<BR><TD> (.*)\n', text)

    for j in range(len(file_name)):
        corpus.append(
            (ROOT_FOLDER + "/" + str(file_name[j]), file_title[j])
        )


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

    return word_tokenize(story)


processed_corpus = []
processed_titles = []


def extract_docs(corpus):
    sample = corpus[:10]
    for path, title in sample:
        file = open(path, 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()
        text = preprocess(text)
        title = preprocess(title)
        processed_corpus.append(text)
        processed_titles.append(title)
    print(processed_titles)


init_corpus()
extract_docs(corpus)
