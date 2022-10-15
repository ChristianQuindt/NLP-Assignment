import re, os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

def readDataset(datapath: str):
    # Get the file details
    directory = []
    file = []
    title = []
    text = []
    label = []
    for dirname, _ , filenames in os.walk(datapath):
        print('Directory: ', dirname)
        print('Subdir: ', dirname.split('/')[-1])

        # traverse direcotries
        for filename in filenames:
            directory.append(dirname)
            file.append(filename)
            label.append(dirname.split('/')[-1])
            fullpathfile = os.path.join(dirname,filename)

            # read files
            with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:
                intext = ''
                firstline = True
                for line in infile:
                    if firstline:
                        title.append(line.replace('\n',''))
                        firstline = False
                    else:
                        intext = intext + ' ' + line.replace('\n','')
                text.append(intext)
    
    # create dataframe    
    fulldf = pd.DataFrame(list(zip(directory, file, title, text, label)), 
        columns =['directory', 'file', 'title', 'text', 'label'])

    return fulldf.filter(['text','label'], axis=1)

def cleanNLFeature(text: str):
    # load symbols and stopwords
    nltk.download('punkt')
    nltk.download("stopwords")
    stop = set(nltk.corpus.stopwords.words("english"))
    s =  set(string.punctuation)

    tok = nltk.word_tokenize(text)
    # remove stopwords, punctuation
    filtered_words = [word.lower() for word in tok if word.lower() not in stop and word.lower() not in s]
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

