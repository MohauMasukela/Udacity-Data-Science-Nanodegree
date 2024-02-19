import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle

import nltk
nltk.download('stopwords',quiet=True)
from nltk.corpus import stopwords

stop_words=stopwords.words('english')


def load_data(database_filename):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database

    """
    engine = create_engine('sqlite:///'+ database_filename)
    conn=engine.connect()
    df= pd.read_sql('select * from DisasterResponse', conn)
    return df


def tokenize(df):
    """
    This function will
    1. lower words
    2. removes special characters
    3. removes stop words
        
    Returns:
    X: Features
    Y: Target
    """

    #lower words
    df['message']=df['message'].apply(lambda x:' '.join([word.lower()
    for word in x.split()]))
    #remove special characters
    df['message']=df['message'].apply(lambda x:' '.join([(re.sub(r"[^a-zA-Z0-9]", " ", word))
    for word in x.split()]))
    #remove stop words
    df['message']=df['message'].apply(lambda x:' '.join([word for word in x.split()
    if word not in (stop_words)]))
    
    X=df["message"]
    y=df.drop(columns =['id', 'message','original','genre'])

    return X,y


def build_model():
     """
    Builds classifier and tunes model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """ 
     pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

     params = {
    'clf__estimator__n_estimators' : [50, 100]
    }
    
     cv = GridSearchCV(pipeline, param_grid=params,cv=3,n_jobs=-1,scoring="accuracy")
    
     return cv   
   


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        df=load_data(database_filename)
        X,y=tokenize(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test,)

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