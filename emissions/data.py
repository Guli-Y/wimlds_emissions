#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 10 2021

"""
import pandas as pd
import numpy as np
from termcolor import colored
from sklearn.model_selection import train_test_split

def load_data(path='../data/sample201320.csv'):
    '''
    1. reads data from data folder
    2. selects relative columns
    3. drops the rows where OVERALL_RESULT is not P or F
    '''
    print(colored("----------------start loading data----------------", 'green'))
    df = pd.read_csv(path, low_memory=False)
    cols = ['TEST_TYPE', 
            'TEST_SDATE',
            'VIN', 
            'VEHICLE_TYPE',
            'MODEL_YEAR',
            'GVWR', 
            'ENGINE_SIZE', 
            'TRANS_TYPE',
            'ODOMETER', 
            'OVERALL_RESULT',
            'MAKE']
    df = df[cols].copy()
    
    # drop the rows where the test was aborted 'A' or overrided 'O'
    df = df[df.OVERALL_RESULT.isin(['P','F'])]
    
    # P=1, F=1
    df['RESULT'] = df['OVERALL_RESULT'].map({'P':0, 'F':1})
    df.drop(columns=['OVERALL_RESULT'], inplace=True)
    print(colored(f"Data loaded: {df.shape[0]} records", 'blue'))
    return df

def clean_data(df):
    '''
    Takes a pandas datafram with at least the following columns: 
        TEST_SDATE, VIN, VEHICLE_TYPE, MODEL_YEAR, ODOMETER, 
        GVWR, ENGINE_SIZE, TRANS_TYPE, TEST_TYPE, MAKE.
    Returns cleaned dataframe
    '''
    print(colored("----------------start cleaning data----------------", 'green'))
        
    # check the data size before cleaning
    print(colored(f'\nRecords in input data: {df.shape[0]}', 'red'))
    print(colored('\nShare of Pass and Fail before cleaning:', 'blue'))
    tmp = 100.0*df['RESULT'].value_counts()/df.shape[0]
    print(colored(f'Fail: {round(tmp[1])}%\nPass: {round(tmp[0])}%', 'blue'))
    print(colored(f"\nUnique vehicles in Fail: {df[df.RESULT==1].VIN.nunique()}",'blue'))
    print(colored(f"Unique vehicles in Pass: {df[df.RESULT==0].VIN.nunique()}",'blue'))
    
    # transform TEST_SDATE to datetime object
    df['TEST_SDATE'] = pd.to_datetime(df['TEST_SDATE'])
    
    # engineering VEHICLE_AGE
    df['VEHICLE_AGE'] = df.TEST_SDATE.dt.year.astype('int') - df.MODEL_YEAR.astype('int') + 2
    
    # get median GVWR for each VIN
    # clean VIN
    df['VIN'] = df['VIN'].astype('string').str.strip().str.lower()
    # get median GVWR
    tmp = df[['VIN', 'GVWR']].groupby('VIN').agg({'GVWR':'median'})
    tmp.reset_index(inplace=True)
    # merge tmp with df
    df = df.merge(tmp, how='left', on='VIN', suffixes=('_0',''))
    # replace 0 with np.nan
    df.loc[df.GVWR==0, 'GVWR'] = np.nan
    df.loc[df.GVWR_0==0, 'GVWR_0'] = np.nan
    # using GVWR_0 fill missing values in GVWR
    df['GVWR'] = df.GVWR.fillna(df.GVWR_0)
    # keep GVW and drop GVWR_0
    df = df.drop(columns=['GVWR_0'])
    # drop na in GVWR
    print('\nRecords with missing GVWR:', df.GVWR.isnull().sum())
    df = df[~df.GVWR.isnull()]
    # drop low numbers in GVWR
    df = df[df.GVWR > 1000]
    print(colored(f'\nRecords after droping rows where GVWR is < 1000 or missing: {df.shape[0]}', 'red'))
    
    # if a vehicle has multiple test records within 90 days, keep earliest record 
    df = df.sort_values('TEST_SDATE')
    df = df.loc[~(df.groupby('VIN')['TEST_SDATE'].diff() < np.timedelta64(90, 'D'))]
    print(colored(f'\nRecords after keeping only the earliest test within a month for each vehicle: {df.shape[0]}', 'red'))
    
    # drop 0s in ODOMETER and remove 9999999 and 8888888
    print('\nRecords where ODOMETER = 0:', df[df.ODOMETER==0].shape[0])
    df = df[(df.ODOMETER!=0) & (df.ODOMETER!=8888888) & (df.ODOMETER!=9999999)]
    print(colored(f'\nRecords after droping rows where ODOMETER is missing: {df.shape[0]}', 'red'))
    # engineer MILE_YEAR from ODOMETER
    df['MILE_YEAR'] = np.round(df['ODOMETER']/df['VEHICLE_AGE'], 2)
    # remove the outliers
    df = df[df.MILE_YEAR <= 40000]
    df = df[~((df.VEHICLE_AGE > 10) & (df.MILE_YEAR < 1000))]
    print(colored(f'\nRecords after droping rows where MILE_YEAR > 40,000: {df.shape[0]}', 'red'))

    # engineer ENGINE_WEIGHT_RATIO
    df['ENGINE_WEIGHT_RATIO'] = np.round(df['ENGINE_SIZE']/df['GVWR'], 2)
      
    #set make to string and to lower case, strip trailing and internal whitespace
    df['MAKE'] = df['MAKE'].astype('string').str.strip().str.lower().str.replace(' ', '')
    
    # select columns
    cols = ['VEHICLE_TYPE',
            'MODEL_YEAR',
            'VEHICLE_AGE',
            'MILE_YEAR',
            'GVWR',
            'ENGINE_SIZE', 
            'TRANS_TYPE', 
            'TEST_TYPE',
            'RESULT',
            'VIN', # will drop this later
            'MAKE',
            'ENGINE_WEIGHT_RATIO',
            'TEST_SDATE'
            ]
    df = df[cols].copy()
    
    # dropnas
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # check data size after cleaning 
    print(colored(f'\nRecords in output data:{df.shape[0]}', 'red'))
    print(colored('\nShare of Pass and Fail after cleaning:', 'blue'))
    tmp = 100.0*df['RESULT'].value_counts()/df.shape[0]
    print(colored(f'Fail: {round(tmp[1])}%\nPass: {round(tmp[0])}%', 'blue'))
    print(colored(f"\nUnique vehicles in Fail: {df[df.RESULT==1].VIN.nunique()}",'blue'))
    print(colored(f"Unique vehicles in Pass: {df[df.RESULT==0].VIN.nunique()}",'blue'))
    # drop VIN
    df = df.drop(columns=['VIN'])
    return df

def split(df=None, test_size=0.2):
    '''
    takes a dataframe and performs a train_test_split
    returns X_train, X_test, y_train, y_test
    '''
    if df is None:   
        df = load_data()   
        df = clean_data(df) 
    X = df.drop(columns=['RESULT'])
    y = df['RESULT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # print some statistics
    print(colored("----------------data splitted into train test----------------", 'green'))
    
    print(colored('Share of Pass and Fail in train set:', 'blue'))
    tmp = 100.0*y_train.value_counts()/y_train.shape[0]
    print(colored(f'Pass: {round(tmp[1])}%\nFail: {round(tmp[0])}%', 'blue'))
    
    print(colored('Share of Pass and Fail in test set:', 'blue'))
    tmp = 100.0*y_test.value_counts()/y_test.shape[0]
    print(colored(f'Pass: {round(tmp[1])}%\nFail: {round(tmp[0])}%', 'blue'))
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    clean_df = clean_data(df)
