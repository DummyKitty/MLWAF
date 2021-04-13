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

classifier_results = pickle.load( open( "data/trained_classifiers_add_F1_score.p", "rb" ) )



def get_classifier_name(index):
    '''
    Returns the name of the classifier at the given index name
    '''
    return index.split()[len(index.split())-1]
    
#Group rows together using same classifier
grouped = classifier_results.groupby(get_classifier_name)

hist_df = pd.DataFrame(columns=['custom','count 1grams','count 2grams','count 3grams','tfidf 1grams','tfidf 2grams','tfidf 3grams'])

for classifier, indices in grouped.groups.items():
    
    #Make a list of feature spaces
    feature_spaces = indices.tolist()
    feature_spaces = [feature_space.replace(classifier,'') for feature_space in feature_spaces]
    feature_spaces = [feature_space.strip() for feature_space in feature_spaces]

    #If no result exists, it will stay as 0
    hist_df.loc[classifier] = {
            'custom':0,
            'count 1grams':0,
            'count 2grams':0,
            'count 3grams':0,
            'tfidf 1grams':0,
            'tfidf 2grams':0,
            'tfidf 3grams':0
    }
    
    #Extract F1-score from classifier_results to corrensponding entry in hist_df
    for fs in feature_spaces:
        hist_df[fs].loc[classifier] = classifier_results['F1-score'].loc[fs + ' ' + classifier]
        

#Plot the bar plot
f, ax = plt.subplots()
ax.set_ylim([0.978,1])
hist_df.plot(kind='bar', figsize=(12,7), title='F1-score of all models grouped by classifiers', ax=ax, width=0.8)
plt.subplots_adjust(bottom=0.205)
# plt.show()
plt.savefig("imgs/models_bar_plot.jpg")

#Make Avgerage F1-score row and cols for the table and print the table
hist_df_nonzero = hist_df.copy()
hist_df_nonzero[hist_df > 0] = True
hist_df['Avg Feature'] = (hist_df.sum(axis=1) / np.array(hist_df_nonzero.sum(axis=1)))
hist_df_nonzero = hist_df.copy()
# hist_df_nonzero[hist_df > 0] = True
hist_df.loc['Avg Classifier'] = (hist_df.sum(axis=0) / np.array(hist_df_nonzero.sum(axis=0)))
hist_df = hist_df.round(4)
display(hist_df)


def plot_learning_curve(df_row,X,Y):
    '''Plots the learning curve of a classifier with its parameters
    
    input - df_row: row of classifier_result
            X: payload data
            Y: labels
    '''
    #The classifier to plot learning curve for
    estimator = df_row['model']
    
    title = 'Learning curves for classifier ' + df_row.name
    train_sizes = np.linspace(0.1,1.0,5)
    cvv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    cv = cvv.split(X)

    #plot settings
    # fig, ax = plt.subplots()
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    print('learning curve in process...')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, n_jobs=-1, train_sizes=train_sizes, verbose=0) #Change verbose=10 to print progress
    print('Learning curve done!')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    # plt.show()
    plt.savefig("imgs/{}.jpg".format(title))

#plot learning curve for tfidf 1grams RandomForest
X = payloads['payload']
Y = payloads['is_malicious']
plot_learning_curve(classifier_results.iloc[0],X,Y)

#plot learning curve for count 3grams MultinomialNB
X = payloads['payload']
Y = payloads['is_malicious']
plot_learning_curve(classifier_results.iloc[6],X,Y)

#plot learning curve for custom svm
X = create_features(pd.DataFrame(payloads['payload'].copy()))
Y = payloads['is_malicious']
plot_learning_curve(classifier_results.iloc[5],X,Y)


def visualize_result(classifier_list):
    '''Plot the ROC curve for a list of classifiers in the same graph
    
    input - classifier_list: a subset of classifier_results
    '''

    f, (ax1, ax2) = plt.subplots(1,2)
    f.set_figheight(6)
    f.set_figwidth(15)
    
    #Subplot 1, ROC curve
    for classifier in classifier_list:
        ax1.plot(classifier['roc']['fpr'], classifier['roc']['tpr'])
        ax1.scatter(1-classifier['specificity'],classifier['sensitivity'], edgecolor='k')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.0])
    ax1.set_title('ROC curve for top3 and bottom3 classifiers')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.grid(True)
    
    #subplot 2, ROC curve zoomed
    for classifier in classifier_list:
        ax2.plot(classifier['roc']['fpr'], classifier['roc']['tpr'])
        ax2.scatter(1-classifier['specificity'],classifier['sensitivity'], edgecolor='k')
    ax2.set_xlim([0, 0.3])
    ax2.set_ylim([0.85, 1.0])
    ax2.set_title('ROC curve for top3 and bottom3 classifiers (Zoomed)')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.grid(True)
    
    #Add further zoom
    left, bottom, width, height = [0.7, 0.27, 0.15, 0.15]
    ax3 = f.add_axes([left, bottom, width, height])
    
    for classifier in classifier_list:
        ax3.plot(classifier['roc']['fpr'], classifier['roc']['tpr'])
        ax3.scatter(1-classifier['specificity'],classifier['sensitivity'], edgecolor='k')
        
    ax3.set_xlim([0, 0.002])
    ax3.set_ylim([0.983, 1.0])
    ax3.set_title('Zoomed even further')
    ax3.grid(True)
    # plt.show()
    plt.savefig("imgs/ROC curve for top3 and bottom3 classifiers.jpg")

indices = [0,1,2, len(classifier_results)-1,len(classifier_results)-2,len(classifier_results)-3]
visualize_result([classifier_results.iloc[index] for index in indices])

