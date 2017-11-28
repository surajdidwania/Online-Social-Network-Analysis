"""
classify.py
"""
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import pickle

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    """
    if keep_internal_punct==True:
        #x = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)',' ',doc.lower()).split()
        x=[]
        str = doc.lower()
        str1 = string.punctuation
        str2 = str1.replace("_","")
        for exp in str.split():
            if (exp.strip(str2)):
                x.append(exp.strip(str2))
    else:
        x = re.sub('\W+', ' ', doc).lower().split()
    return np.array(x)

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    """
    
    result = Counter(tokens)
    for token in tokens:
        feats.update({'token='+token:result[token]})

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    pass
    elements=[]
    subarray = [sublist for sublist in (tokens[x:x+k] for x in range(len(tokens) - k + 1))]
    for i in range(len(subarray)):
        for j in range(len(subarray[i])):
            for k in range(j):
                element = "token_pair="+subarray[i][k]+'__'+subarray[i][j]
                elements.append(element)
    result = Counter(elements)
    for token in result.keys():
        feats[token] = result[token]

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    pass
    result = []
    feats = defaultdict(lambda: 0)
    for feature in feature_fns:
        feature(tokens,feats)
    return sorted(feats.items())

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]])
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    pass
    result = []
    vocabulary = {}
    count = defaultdict(int)
    indptr = [0]
    indices = []
    data = []
    for tokens in tokens_list:
        feature_val = featurize(tokens,feature_fns)
        result.append(feature_val)
        for res in feature_val:
            count[res[0]]+=1
            vocabulary.setdefault(res[0], len(vocabulary)) 
    if vocab == None:
        vocab = {}
        for element in sorted(vocabulary):
            vocab.setdefault(element,len(vocab))      
        for f in result:
            for term in f:
                if term[0] in vocab:
                    index = vocab[term[0]]
                    indices.append(index)
                    data.append(term[1])
            indptr.append(len(indices))
    else: 
        for res in result:
            finallist=[]  
            for term in res: 
                if term[0] in vocab:
                    index = vocab[term[0]]
                    finallist.append(term)  
                    indices.append(index)
                    data.append(term[1])
            for term in vocab.keys():
                if term not in finallist:
                    index = vocab[term]  
                    indices.append(index)
                    data.append(0)
            indptr.append(len(indices)) 
    matrix=csr_matrix((data, indices, indptr), dtype=np.int64)
    return matrix,vocab

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    pass
    cv = KFold(len(labels),n_folds=k)
    accuracies = []
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    combine = []
    result = []
    combinationlist=[]
    finallist = []   
    for L in range(1, len(feature_fns)+1):
        comblist = [list(subset) for subset in combinations(feature_fns, L)]
        combine.append(comblist)
    for val in combine:
              for s in val:    
                        combinationlist.append(s)
    for punct in punct_vals:
        tokens_list = [tokenize(d,punct) for d in docs]
        for s in combinationlist:
            for freq in min_freqs:
                clf = LogisticRegression()
                X, vocab = vectorize(tokens_list,s,freq)
                avg = cross_validation_accuracy(clf,X,labels,k = 5)
                finaldict={}
                finaldict.update({'accuracy':avg})     
                finaldict.update({'punct':punct})
                finaldict.update({'min_freq':freq})       
                finaldict.update({'features':tuple(s)})     
                finallist.append(finaldict)   
    value = sorted(finallist, key=lambda k: (-k['accuracy']))
    return value  

def train_data(followerdetails, male_names,female_names):
    malecount=0
    femalecount=0
    unknowncount=0
    for follow_details in followerdetails.keys():
        followername = followerdetails[follow_details]['name']
        if followername:
            name = re.findall('\w+',followername.split()[0].lower())
            if(len(name)>0):
                if name[0].lower() in male_names:
                    malecount+=1
                    followerdetails[follow_details]['gender']='male'
                elif name[0].lower() in female_names:
                    femalecount+=1
                    followerdetails[follow_details]['gender']='female'
                else:
                    unknowncount+=1
                    followerdetails[follow_details]['gender']='unknown'
    if (malecount==0):
        for follow_details in followerdetails.keys():
            if (followerdetails[follow_details]['gender']=='unknown'):
                followerdetails[follow_details]['gender']='male'
                malecoun+=1
                unknowncount-=1
                break
    if (femalecount==0):
        for follow_details in followerdetails.keys():
            if (followerdetails[follow_details]['gender']=='unknown'):
                followerdetails[follow_details]['gender']='female'
                femalecount+=1
                unknowncount-=1
                break
            

def document(followerdetails):
    docs = []
    labels = []
    for follow_details in followerdetails.keys():
        if ('gender' in followerdetails[follow_details]):   
            if (followerdetails[follow_details]['gender']!='unknown'):
                docs.append(followerdetails[follow_details]['description'])
                if (followerdetails[follow_details]['gender']=='male'):
                    labels.append(1)
                else:
                    labels.append(0)
    return docs,np.array(labels)

def testing_doc(followerDetails):
    docs = []
    labels = []
    usernameList = []
    for follow_details in followerDetails.keys():
        if ('gender' in followerDetails[follow_details]):   
            if (followerDetails[follow_details]['gender']=='unknown'):
                docs.append(followerDetails[follow_details]['description'])
                usernameList.append(follow_details)
    return docs,usernameList

def predict_testingdoc(best_result,doc,docs,labels,followerDetails,namelist):
    accuracy = best_result['accuracy']
    punct = best_result['punct']
    min_freq = best_result['min_freq']
    features = best_result['features']
    clf = LogisticRegression()
    tokens_trainlist=[]
    tokens_testlist=[]
    for d in docs:
        tokens_trainlist.append(tokenize(d,punct)) 
    for d in doc:
        tokens_testlist.append(tokenize(d,punct))
    X, vocab = vectorize(tokens_trainlist,features,min_freq)
    test_CSR, test_vocab = vectorize(tokens_testlist,features,min_freq,vocab)
    clf.fit(X,labels)
    predictions = clf.predict(test_CSR)
    for user,p in zip(namelist,predictions):
        det = followerDetails[user]
        if p == 1:
            det['gender'] = 'male'
        else:
            det['gender'] = 'female'
        followerDetails[user] = det
def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    pass
    acc = []
    results = sorted(results, key=lambda k: k['accuracy'])
    for result in results:
        acc.append(result['accuracy'])
    plt.ylabel('accuracies')
    plt.xlabel('Feature Functions')
    plt.plot(acc)
    plt.savefig('accuracies.png')

def main():
    feature_fns = [token_features, token_pair_features]
    if not os.path.isfile("users.pkl"):
        print ("data not loaded properly")
    else:
        if (os.path.getsize("users.pkl")==0):
            print ("data size not proper")
        else:
            users = pickle.load(open("users.pkl","rb"))
            friendlist = pickle.load(open("friendlist.pkl","rb"))
            followerdetails = pickle.load(open("followerdetails.pkl","rb"))
            male_names=pickle.load(open("maleNames.pkl","rb"))
            female_names = pickle.load(open("femaleNames.pkl","rb"))
    train_data(followerdetails,male_names,female_names)
    docs,labels = document(followerdetails)
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])

    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    # Fit best classifier.
    doc , namelist = testing_doc(followerdetails)
    predict_testingdoc(best_result,doc,docs,labels,followerdetails,namelist)
    pickle.dump(followerdetails, open('followerdetails.pkl', 'wb'))
if __name__ == '__main__':
    main()