import env
import acquire as a
import pandas as pd
import matplotlib as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



def telco_pre_split_clean(df):
    '''
    function takes in the Telco DataFrame from the acquire module's get_telco_data function and applies steps to drop duplicate and unnecessary columns as well a casting the total_charges feature from object to float values.
    '''
# there are many duplicate columns all of which end in a 1 or 2
# there are no other columns that end in a number.
# drop duplicate columns with same names:
    df = df.T.drop_duplicates().T
# I will create a list of duplicate columns with a loop 
# then I will feed that list to .drop()
    dup_cols = [col for col in df.columns if col.endswith(('1', '2'))]
    df.drop(columns=dup_cols, inplace=True)
# drop columns contract_type_id, internet_service_type_id, 
# and payment_type_id as they are redundant and I will
# encode their more verbose counterparts prior to modeling.
    df.drop(columns=['contract_type_id', 'internet_service_type_id',
                 'payment_type_id'], inplace=True)
# most of signup_date usefulness is already reflected in tenure
# so choosing to drop signup_date
    df.drop(columns='signup_date', inplace=True)
# customer_id field is irrelavent after importing to a DataFrame 
# and could be detrimental during modeling so will drop.
    df.drop(columns='churn_month', inplace=True)
# need to cast total_charges to float.
# there are empty values for new customers which cannot be cast
# adding zero to all values fixes problem without changing values
    df.total_charges = (df.total_charges + '0').astype('float')
    return df

def telco_encoded_cleaned(df):
    '''
    function takes in the output of the telco_pre_split_clean
    and encodes all of the useful features for (pre-split) for (post-split)
    ML analysis.
    '''
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    df = pd.concat( [df, dummy_df], axis=1 )
    return df

def train_validate_test_split(df, target='churn', seed=9751):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


def wrangle_telco():
    train, val, test = train_validate_test_split(
        telco_encoded_cleaned(
            telco_pre_split_clean(
                a.get_telco_data())))
        
    return train, val, test