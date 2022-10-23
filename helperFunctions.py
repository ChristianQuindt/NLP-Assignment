import re, os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import ngrams

def preprocessNLFeature(text: str):
    # load symbols and stopwords
    nltk.download('punkt')
    nltk.download("stopwords")
    stop = set(nltk.corpus.stopwords.words("english"))
    s =  set(string.punctuation)

    tokens = nltk.word_tokenize(text)
    # remove stopwords, punctuation
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop and word.lower() not in s]
    # remove digits
    filtered_words = [word.replace('\d+', '') for word in filtered_words]
    # remove one-char tokens
    filtered_words = [word for word in filtered_words if len(word) > 1]

    text = ' '.join(word for word in filtered_words)
    # remove digits
    text.replace('\d+', '')
    
    return text

def createNgrams(text: str, n):
    return list(ngrams(text.split(), n))