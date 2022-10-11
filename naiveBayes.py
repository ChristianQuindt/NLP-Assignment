import numpy as np
from pandas import DataFrame, Series

# returns prediction of truth of a event(word) following a specified hypothesis(class).
# If hypothesis is None, the probabilities for all hypothesies preceding the event will be calculated
def predict(uniques: list, freq_tb: np.ndarray, sumRowsRel: np.ndarray, sumColsRel: np.ndarray, event: str, hypothesis: np.int64 = None ):
        if event not in uniques:
                #print(event + ' not in vocabulary!')
                return
        if hypothesis == None:
                return (predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, 0),
                        predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, 1),
                        predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, 2),
                        predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, 3),
                        predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, 4))
        # P(hypothesis|event) = (P(event|hypothesis) * P(hypothesis)) / P(event)
        i = uniques.index(event)
        p_h_e = freq_tb[i][hypothesis] / np.sum(freq_tb, axis=0)[hypothesis]
        p_h = sumRowsRel[i]
        p_e = sumColsRel[hypothesis]
        return p_h_e * p_h / p_e

def predictDoc(document: str, uniques: list, freq_tb: np.ndarray, sumRowsRel: np.ndarray, sumColsRel: np.ndarray):
    # probabilities for a word to belong to a class (for every class)
    documentsProbs = []
    for word in document.split():
        p = predict(uniques, freq_tb, sumRowsRel, sumColsRel, word)
        if p is not None:
            documentsProbs.append( [i for i in p] ) 
    
    # sum up the probabilities
    return np.argmax(np.sum(documentsProbs, axis=0))

def frequencyTable(df: DataFrame):
    # get the complete vocabulary
    stack = df['text'].str.split(' ', expand=True).stack()
    uniques = list(stack.unique())

    # count words in a document
    counts = df['text'].apply(lambda x: np.unique(x.split(' '), return_counts=True))
    # init frequency table with vocab in rows and labels as columns
    freq_tb = np.zeros((len(uniques), df.label.nunique()))

    # iterate through all documents word counts and fill the freq table
    # foreach document
    for i in range(len(counts)):
        # foreach word in document
        for j in range(len(counts[i][0])):
            # word exists in vocabulary (may not be the case due to train val test split)
            if counts[i][0][j] in uniques:
                # update table on corresponding index [wordindex][labelindex]
                freq_tb[uniques.index(counts[i][0][j])][df.label[i]] += counts[i][1][j]
    return freq_tb, uniques

def likelihoodTable(freq_tb: np.array):
    # total, row and columns wordcount
    sumTotal = np.sum(freq_tb)
    sumRows = np.sum(freq_tb, axis=1) 
    sumCols = np.sum(freq_tb, axis=0)
    
    sumRowsRel = sumRows/sumTotal
    sumColsRel = sumCols/sumTotal

    return sumRowsRel, sumColsRel


