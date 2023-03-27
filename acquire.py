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

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers c
	left join customer_payments using(customer_id)
    left join customer_churn using(customer_id)
    left join customer_contracts using(customer_id)
    left join customer_details using(customer_id)
    left join customer_signups using(customer_id)
    left join customer_subscriptions using(customer_id)
    left join internet_service_types as ist 
		on c.internet_service_type_id = ist.internet_service_type_id
    left join contract_types as ct
		on c.contract_type_id = ct.contract_type_id
	left join payment_types as pt
		on c.payment_type_id = pt.payment_type_id
;
                 """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df
