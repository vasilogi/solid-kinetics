# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local application imports
from .arrhenius import  mass2conv

def get_data(DATA_DIR):
    # get the csv data in a list
    return [os.path.join(DATA_DIR,f) for f in os.listdir(DATA_DIR) if 'csv' in f]


def read_filtrated_datafile(df,low,high):
    # df   : dataframe
    # low  : lower limit of conversion
    # high : higher limit of conversion

    # Check if conversion columns exists in the CSV file

    if 'conversion' in df:
        # filter data
        rangeFilter  = (df['conversion']>= low) & (df['conversion']<= high)
        df           = df[rangeFilter]
    else:
        # convert mass to conversion
        mass = df['mass'].to_numpy()
        m0   = mass[0]    # initial weight
        minf = mass[-1]   # final weight
        conversion = np.array([mass2conv(m0,mt,minf) for mt in mass])
        df['conversion'] = conversion
        # filter data
        rangeFilter  = (df['conversion']>= low) & (df['conversion']<= high)
        df           = df[rangeFilter]

    conversion   = df['conversion'].to_numpy()
    time         = df['time'].to_numpy()
    temperature  = df['temperature'].to_numpy()[0]
        
    return conversion, time, temperature

def read_units(df):
    # df   : dataframe
    timeUnits = df['time units'].to_list()[0]
    tempUnits = df['temperature units'].to_list()[0]

    if tempUnits == 'Kelvin':
        tempUnits = 'K'
    elif tempUnits == 'Celsius':
        tempUnits = 'C'
    
    return timeUnits, tempUnits