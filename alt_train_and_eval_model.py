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


column_to_predict = "ticket_type"
# Supported datasets:
# ticket_type
# business_service
# category
# impact
# urgency
# sub_category1
# sub_category2

classifier = "NB"  # Supported algorithms # "SVM" # "NB"
use_grid_search = True  # grid search is used to find hyperparameters. Searching for hyperparameters is time consuming
remove_stop_words = True  # removes stop words from processed text
stop_words_lang = 'english'  
use_stemming = True  # word stemming using nltk
fit_prior = False  # if use_stemming == True then it should be set to False ?? double check
min_data_per_class = 1  # used to determine number of samples required for each class.Classes with less than that will be excluded from the dataset.

if __name__ == '__main__':

     
    # loading dataset from csv
    dfTickets = pd.read_csv(
        './support-tickets-classification/datasets/all_tickets.csv',
        dtype=str
    )  

    text_columns = "body"  # "title" 
    
    # Removing rows related to classes represented by low amount of data
    print("Shape of dataset before removing classes with less then " + str(min_data_per_class) + " rows: "+str(dfTickets.shape))
    print("Number of classes before removing classes with less then " + str(min_data_per_class) + " rows: "+str(len(np.unique(dfTickets[column_to_predict]))))
    bytag = dfTickets.groupby(column_to_predict).aggregate(np.count_nonzero)
    tags = bytag[bytag.body > min_data_per_class].index
    dfTickets = dfTickets[dfTickets[column_to_predict].isin(tags)]
    print(
        "Shape of dataset after removing classes with less then "
        + str(min_data_per_class) + " rows: "
        + str(dfTickets.shape)
    )
    print(
        "Number of classes after removing classes with less then "
        + str(min_data_per_class) + " rows: "
        + str(len(np.unique(dfTickets[column_to_predict])))
    )

    labelData = dfTickets[column_to_predict]
    data = dfTickets[text_columns]

    # Split dataset into training and testing data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labelData, test_size=0.2
    )  # split data to train/test sets with 80:20 ratio

    # Extracting features from text
    # Count vectorizer
    if remove_stop_words:
        count_vect = CountVectorizer(stop_words=stop_words_lang)
    elif use_stemming:
        count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    else:
        count_vect = CountVectorizer()

    # Fitting the training data into a data processing pipeline and eventually into the model itself
    if classifier == "NB":
        print("Training NB classifier")
        # Building a pipeline

        text_clf = Pipeline([
            ('vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB(fit_prior=fit_prior))
        ])
        text_clf = text_clf.fit(train_data, train_labels)

    elif classifier == "SVM":
        print("Training SVM classifier")
        # Training Support Vector Machines - SVM
        text_clf = Pipeline([(
            'vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                loss='hinge', penalty='l2', alpha=1e-3,
                n_iter=5, random_state=42
            )
        )])
        text_clf = text_clf.fit(train_data, train_labels)

    if use_grid_search:
        # NB parameters
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }

        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, train_labels)

       
        gs_clf.best_score_
        gs_clf.best_params_

    print("Evaluating model")
    # Score and evaluate model on test data using model without hyperparameter tuning
    predicted = text_clf.predict(test_data)
    prediction_acc = np.mean(predicted == test_labels)
    print("Confusion matrix without GridSearch:")
    print(metrics.confusion_matrix(test_labels, predicted))
    print("Mean without GridSearch: " + str(prediction_acc))

    # Score and evaluate model on test data using model WITH hyperparameter tuning
    if use_grid_search:
        predicted = gs_clf.predict(test_data)
        prediction_acc = np.mean(predicted == test_labels)
        print("Confusion matrix with GridSearch:")
        print(metrics.confusion_matrix(test_labels, predicted))
        print("Mean with GridSearch: " + str(prediction_acc))

    # Ploting confusion matrix with 'seaborn' module
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import matplotlib
    mat = confusion_matrix(test_labels, predicted)
    plt.figure(figsize=(4, 4))
    sns.set()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    # plt.savefig(os.path.join('.', 'outputs', 'confusion_matrix.png'))
    plt.show()

    # Printing classification report
    from sklearn.metrics import classification_report
    print(classification_report(test_labels, predicted,
                                target_names=np.unique(test_labels)))

    # Save trained models to /output folder
    if use_grid_search:
        pickle.dump(
            gs_clf,
            open(  "support-tickets-classification/outputs/ticket_type.sav",
                'wb'
            )
        )
    else:
        pickle.dump(
            text_clf,
            open(  "support-tickets-classification/outputs/ticket_type.sav",
                'wb'
            )
        )
