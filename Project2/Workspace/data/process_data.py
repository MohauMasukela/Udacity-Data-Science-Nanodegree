

import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges datasets from 2 filepaths.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Returns:
    df: dataframe containing messages_filepath and categories_filepath merged
    
    """
    print("Loading Database.........")
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df =  pd.merge(categories,messages,on='id')
    return df



def clean_data(df):

    categories = df["categories"].str.split(";",expand=True)
    row=categories[:1]
    category_colnames=row.iloc[0]
    category_colnames=category_colnames.str.split("-").str[0:1].str.join("")
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column]=categories[column].str[-1]

        # set each value to be the last character of the string
        categories[column]=categories[column].astype(int)
    df=df.drop(["categories"], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    df=df.drop_duplicates()
    #remove non binary 
    df = df[df["related"]!=2]
    print("Database Cleaned........")
    return df


def save_data(df, database_filename):
      
    """ This function stores the df in a SQL database """

    print("Creating Database.....")

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)



def main():
        """
        Data processing functions:
           1) Load Messages Data with Categories
           2) Clean Categories Data
           3) Save Data to SQLite Database
       """
        if len(sys.argv) == 4:

            messages_filepath, categories_filepath,database_filename = sys.argv[1:]
            df = load_data(messages_filepath, categories_filepath)
            df = clean_data(df)
            save_data(df, database_filename)
            print(df)

        else:
            print('Please provide the filepaths of the messages and categories '\
                'datasets as the first and second argument respectively, as '\
                'well as the filepath of the database to save the cleaned data '\
                'to as the third argument. \n\nExample: python process_data.py '\
                'disaster_messages.csv disaster_categories.csv '\
                'DisasterResponse.db')
 


if __name__ == '__main__':
    main()