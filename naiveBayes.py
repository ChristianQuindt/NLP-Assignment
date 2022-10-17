import numpy as np
from pandas import DataFrame, Series


# returns prediction of truth of an event(instance to be classified) following a specified hypothesis(class).
# If hypothesis is None, the probabilities for all hypothesies preceding the event will be calculated
def predict(uniques: list, freq_tb: np.ndarray, sumRowsRel: np.ndarray, sumColsRel: np.ndarray, event, hypothesis: np.int64 = None, classes: range = range(2)):
        if event not in uniques:
                #print(event + ' not in vocabulary!')
                return
        if hypothesis == None:
                return (predict(uniques, freq_tb, sumRowsRel, sumColsRel, event, i) for i in classes)
        # P(hypothesis|event) = (P(event|hypothesis) * P(hypothesis)) / P(event)
        i = uniques.index(event)
        p_h_e = freq_tb[i][hypothesis] / np.sum(freq_tb, axis=0)[hypothesis]
        p_h = sumRowsRel[i]
        p_e = sumColsRel[hypothesis]
        return p_h_e * p_h / p_e

def predictDoc(document: np.iterable, uniques: list, freq_tb: np.ndarray, sumRowsRel: np.ndarray, sumColsRel: np.ndarray):
    # probabilities for an instance to classify to belong to a class (for every class)
    documentsProbs = []
    for iter in document:
        p = predict(uniques, freq_tb, sumRowsRel, sumColsRel, iter)
        if p is not None:
            documentsProbs.append( [i for i in p] ) 
    
    # empty list, no prediction could be made
    if not documentsProbs:
        return -1
    # sum up the probabilities
    return np.argmax(np.sum(documentsProbs, axis=0))

def frequencyTableOccurences(toCount: Series, labels: Series):
    # get the complete vocabulary
    stack = toCount.str.split(' ', expand=True).stack()
    uniques = list(stack.unique())

    # count words in a document
    counts = toCount.apply(lambda x: np.unique(x.split(' '), return_counts=True))
    
    # sadly, the indices don't match between the NB variants -> code replication with altered indices
    # init frequency table with vocab in rows and labels as columns
    freq_tb = np.zeros((len(uniques), labels.nunique()))
    
    # iterate through all documents instances to count and fill the freq table
    # foreach document
    for i in range(len(counts)):
        # foreach instance to count in the document
        for j in range(len(counts[i][0])):
            # instance to count exists in vocabulary (may not be the case due to train val test split)
            if counts[i][0][j] in uniques:
                # update table on corresponding index [instances-to-count-index][labelindex]
                freq_tb[uniques.index(counts[i][0][j])][labels[i]] += counts[i][1][j]
    return freq_tb, uniques
    
def frequencyTableNgrams(toCount: Series, labels: Series):
    # get the complete vocabulary
    uniques = set()
    for ngramlst in toCount[:5]:
        for st in ngramlst:
            uniques.add(st)
    uniques = list(uniques)

    # count occurance of each n-gram of a document
    counts = toCount.apply(lambda lstOfSets: [(_set, lstOfSets.count(_set)) for _set in sorted(set(lstOfSets))])

    # sadly, the indices don't match between the NB variants -> code replication with altered indices
    freq_tb = np.zeros((len(uniques), 2))
    # iterate through all documents instances to count and fill the freq table
    # foreach document
    for i in range(len(counts)):
        # foreach instance to count in the document
        for j in range(len(counts[i])):
            # instance to count exists in vocabulary (may not be the case due to train val test split)
            if counts[i][j][0] in uniques:
                # update table on corresponding index [instances-to-count-index][labelindex]
                toAdd = counts[i][j][1]
                freq_tb[uniques.index(counts[i][j][0])][labels[i]] += toAdd

    return freq_tb, uniques    

def likelihoodTable(freq_tb: np.array):
    # total, row and columns wordcount
    sumTotal = np.sum(freq_tb)
    sumRows = np.sum(freq_tb, axis=1) 
    sumCols = np.sum(freq_tb, axis=0)
    
    sumRowsRel = sumRows/sumTotal
    sumColsRel = sumCols/sumTotal

    return sumRowsRel, sumColsRel


