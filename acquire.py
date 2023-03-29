import pandas as pd
import numpy as np
import os
from env import host, user, password


def get_connection(db, u=user, h=host, p=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{u}:{p}@{h}/{db}'

def new_zillow_data2016():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                select bedroomcnt, bathroomcnt, 
                calculatedfinishedsquarefeet, taxvaluedollarcnt, 
                yearbuilt, taxamount, fips from properties_2016
                where propertylandusetypeid = 261
                ;

                 """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def new_zillow_data2017():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                select bedroomcnt, bathroomcnt, 
                calculatedfinishedsquarefeet, taxvaluedollarcnt, 
                yearbuilt, taxamount, fips from properties_2017
                where propertylandusetypeid = 261
                ;

                 """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data2016():
    '''
    This function reads in zillow 2016 data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow2016.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow2016.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data2016()
        
        # Cache data
        df.to_csv('zillow2016.csv')
        
    return df


def get_zillow_data2017():
    '''
    This function reads in zillow 2017 data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow2017.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow2017.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data2017()
        
        # Cache data
        df.to_csv('zillow2017.csv')
        
    return df