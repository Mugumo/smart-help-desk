import sys
import nltk
import numpy as np
import pandas as pd
import pickle
# from helpers import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
from matplotlib import pyplot as plt
sys.path.append(".")
sys.path.append("..")


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])



remove_stop_words = True
stop_words_lang = 'english'
use_stemming = True 
fit_prior = False 

labelData = dfTickets[column_to_predict]

dfTickets = pd.read_csv(
        'support-tickets-classification/test.csv',
        dtype=str
    )  
data = dfTickets[text_columns]


count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)


text_clf = Pipeline([
            ('vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB(fit_prior=fit_prior))
        ])
text_clf = text_clf.fit(train_data, train_labels)

#generate predictions
predicted = text_clf.predict(test_data)

#store predictions in csv file
mnb_results = np.array(list(zip(test_df['id'],predicted)))
mnb_results = pd.DataFrame(mnb_results, columns=['issue', 'type'])
mnb_results.to_csv('results.csv', index = False)