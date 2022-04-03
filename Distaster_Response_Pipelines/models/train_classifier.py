import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
import pickle
import argparse

def load_data(database_filepath):
    '''
    Load data from database

    Args:
    - database_filepath (str)

    Returns:
    - X (pandas.Series) : dataset
    - Y (pandas.DataFrame) : categories
    - category_names (list) : list of Category Names

    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    '''
    Clean and tokenize the text in input
    Args:
        text (str): input text
    Returns:
        clean_tokens (list): tokens obtained from the input text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build the machine learning model pipeline

    Returns:
    - model_cv : model

    '''
    # create the model pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Use grid search to find better parameters
    parameters = {
    #'vect__ngram_range': ((1, 1), (1, 2)), 
    #'vect__max_df': (0.5, 0.75, 1.0), 
    #'tfidf__use_idf': (True, False), 
    #'clf__estimator__n_estimators': [50, 100, 200], 
    'clf__estimator__min_samples_split': [2, 3, 4]
    }
    model_cv = GridSearchCV(pipeline, param_grid = parameters)

    return model_cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances and print the results
    
    Args:
        model : model to evaluate
        X_test : dataset
        Y_test : dataframe containing the categories
        category_names (str): categories name
    '''

    Y_pred = model.predict(X_test)

    # Calculate the accuracy for each category.
    for i in range(len(category_names)):
       print('Category: {} '.format(category_names[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filepath):
    '''
    Save model in pickle file
    
    Args:
        model : model to be saved
        model_filepath (str): destination pickle filename
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()