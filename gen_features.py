import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn
import string
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


import sklearn.gaussian_process.kernels as kernels

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from scipy.stats import expon



def get1Grams(payload_obj):
    '''Divides a string into 1-grams
    
    Example: input - payload: "<script>"
             output- ["<","s","c","r","i","p","t",">"]
    '''
    payload = str(payload_obj)
    ngrams = []
    for i in range(0,len(payload)-1):
        ngrams.append(payload[i:i+1])
    return ngrams


def get2Grams(payload_obj):
    '''Divides a string into 2-grams
    
    Example: input - payload: "<script>"
             output- ["<s","sc","cr","ri","ip","pt","t>"]
    '''
    payload = str(payload_obj)
    ngrams = []
    for i in range(0,len(payload)-2):
        ngrams.append(payload[i:i+2])
    return ngrams


def get3Grams(payload_obj):
    '''Divides a string into 3-grams
    
    Example: input - payload: "<script>"
             output- ["<sc","scr","cri","rip","ipt","pt>"]
    '''
    payload = str(payload_obj)
    ngrams = []
    for i in range(0,len(payload)-3):
        ngrams.append(payload[i:i+3])
    return ngrams


def visualize_feature_space_by_projection(X,Y,title='PCA'):
    '''Plot a two-dimensional projection of the dataset in the specified feature space
    
    input: X - data
           Y - labels
           title - title of plot
    '''
    pca = TruncatedSVD(n_components=2)
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['blue', 'darkorange']
    lw = 2

    #Plot malicious and non-malicious separately with different colors
    for color, i, y in zip(colors, [0, 1], Y):
        plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.3, lw=lw,
                    label=i)
        
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.savefig("imgs/{}.jpg".format(title))


payloads = pd.read_pickle("data/payloads.pkl")

tfidf_vectorizer_1grams = TfidfVectorizer(tokenizer=get1Grams)
count_vectorizer_1grams = CountVectorizer(min_df=1, tokenizer=get1Grams)

tfidf_vectorizer_2grams = TfidfVectorizer(tokenizer=get2Grams)
count_vectorizer_2grams = CountVectorizer(min_df=1, tokenizer=get2Grams)

tfidf_vectorizer_3grams = TfidfVectorizer(tokenizer=get3Grams)
count_vectorizer_3grams = CountVectorizer(min_df=1, tokenizer=get3Grams)




if __name__ == "__main__":
    


    X = count_vectorizer_1grams.fit_transform(payloads['payload'])
    Y = payloads['is_malicious']
    visualize_feature_space_by_projection(X,Y,title='PCA visualization of 1-grams CountVectorizer feature space')

    X = tfidf_vectorizer_2grams.fit_transform(payloads['payload'])
    Y = payloads['is_malicious']
    visualize_feature_space_by_projection(X,Y,title='PCA visualization of 2-grams TFIDFVectorizer feature space')


    X = tfidf_vectorizer_3grams.fit_transform(payloads['payload'])
    Y = payloads['is_malicious']
    visualize_feature_space_by_projection(X,Y,title='PCA visualization of 3-grams TFIDFVectorizer feature space')

    X = create_features(pd.DataFrame(payloads['payload'].copy()))
    Y = payloads['is_malicious']
    visualize_feature_space_by_projection(X,Y,title='PCA visualization of custom feature space')
