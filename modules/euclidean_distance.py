import os
import pandas as pd
import numpy as np

# euclidean distance in the context of model fitting method (COMF)
# https://www.mdpi.com/2073-4352/10/2/139
def d(p,q,r):
    return np.sqrt((0.0-p)**2.0 + (0.0-q)**2.0 + (0.0-r)**2.0)

def convergenceData(df):
    # arguments: a dataframe that contains at least the columns | models | MSE - alpha | MSE - integral
    # returns  : a pandas dataframe containing the model with the fitting error (Euclidean Distance)
    
    # filter the negative determination coefficient values
    numDf = df._get_numeric_data()
    numDf[numDf < 0] = 0

    # calculate the Euclidean distance (fitting error)
    p = numDf['MSE - alpha'].to_numpy()
    q = numDf['MSE - integral'].to_numpy()
    r = numDf['MSE - differential'].to_numpy()
    dist = d(p,q,r)
    data = {'model': df['model'].to_list(), 'fitting error': dist}
    return pd.DataFrame(data)