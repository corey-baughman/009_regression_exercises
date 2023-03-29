import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
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
                calculatedfinishedsquarefeet,
                taxvaluedollarcnt, yearbuilt, 
                taxamount, fips 
                from properties_2017 
                where propertylandusetypeid = 261 or 
                propertylandusetypeid = 279 
                limit 50;
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


# There are a fair number of extreme outliers in all features except for fips.
# The ones in year may still be useful as age of houses do not necessarily
# dramatically change their value. Since we are trying to model home prices
# across typical home values, I will drop the outliers except for year. 
#The outliers, especially on the high end will skew the model and 
#realistically they probably need their own models.

col_list = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount']


def clean_zillow_data2017():
    '''
    This function retrieves the zillow data from the CodeUp MySQL database
    and applies cleaning steps to drop observations with null values,
    reset the index after dropping rows, and cast bedroomcnt, yearbuilt, 
    and fips to integers. It returns the wrangled dataframe. Function relies
    on other functions in the wrangle.py module.
    '''
    df = get_zillow_data2017()
    # standardize column names to something more pythonic
    df = df.rename(columns = {'bedroomcnt' : 'bedrooms', 
                              'bathroomcnt' : 'bathrooms', 
                              'calculatedfinishedsquarefeet' :'area', 
                              'taxvaluedollarcnt' : 'tax_value', 
                              'yearbuilt' : 'year_built', 
                              'taxamount' : 'tax_amount'})
    # dropping all nulls as they are less than 1% of observations
    # and scattered across the features.
    df = df.dropna()
    # may as well reset the index after dropping nulls
    df.reset_index(drop=True, inplace=True)
    # bedrooms, year built, and fips code should be integers
    df.bedrooms = df.bedrooms.astype(int)
    df.year_built = df.year_built.astype(int)
    df.fips = df.fips.astype(object)
    
    return df


def remove_outliers(df, col_list=col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns
    using the tukey method.
    return a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    df.reset_index(drop=True, inplace=True)
    
    return df



def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; 
    Return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=9751)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=9751)
    return train, validate, test



def wrangle_zillow():
    '''
    This function retrieves the zillow data from the CodeUp MySQL database
    and applies cleaning steps to drop observations with null values,
    reset the index after dropping rows, and cast bedrooms, year_built, 
    and fips to integers. Then it removes outliers using the tukey method and finally splits the dataframe into Train, Validate, 
    and Test dataframes. Function relies
    on other functions in the wrangle.py module.
    
    Arguments: None
    Returns: Train, Validate, Test dataframes
    '''
    
    train, validate, test = split_data(remove_outliers(clean_zillow_data2017()))
    
    
    return train, validate, test
    
    