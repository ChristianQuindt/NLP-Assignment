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
    """ # lower case
    text = text.lower()
    
    # regex with some symbols that (might) need to be replaced/removed manually 
    TO_SPACE = re.compile('[/(){}\[\]\|@,;]')
    REMOVE = re.compile('[^0-9 ^0-9a-z ^0-9+ #+_]')
    
    # replace symbols by space in text
    text = TO_SPACE.sub(' ', text)
    
    # remove symbols completely
    text = REMOVE.sub('', text) 
    
    # tokenize for smaller vocabulary
    tok = nltk.word_tokenize(text)
    filtered_words = [word for word in tok if word not in stop and word not in s]
    
    # remove non-word character
    filtered_words = [re.sub(r'\W+', '', word) for word in filtered_words]
    
    text = ' '.join(word for word in filtered_words)  """
    return text

def createNgrams(text: str, n):
    return list(ngrams(text.split(), n))