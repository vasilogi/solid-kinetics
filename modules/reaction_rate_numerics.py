# Standard library imports
import os

# Third party imports
from scipy.misc import derivative
from scipy import interpolate
import pandas as pd
import numpy as np

# Local application imports
from modules.file_handlers import read_filtrated_datafile, get_data
# from modules.file_handlers import read_filtrated_datafile, get_data, read_units

def data2PolFit(Csv,low,high,pdeg,npoints):
    # actual experimental conversion to polynomial fit
    # pdeg: degree of polynomial
    
    # get dataframe
    df = pd.read_csv(Csv)
    # data
    conversion, time, temperature = read_filtrated_datafile(df,low,high)
    # polynomial fit
    z          = np.polyfit(time,conversion,pdeg)
    polynomial = np.poly1d(z)
    t_polfit   = time
    # t_polfit   = np.linspace(time[0],time[-1],1000) # interpolate to these new points
    a_polfit   = polynomial(t_polfit)

    return a_polfit

def data2Polrate(Csv,low,high,pdeg,npoints):
    # actual experimental conversion to polynomial fit
    # then derivative of the polynomial fit on npoints new time data
    # then interpolate of the derivative on original time data
    # pdeg: degree of polynomial
    
    # get dataframe
    df = pd.read_csv(Csv)
    # data
    conversion, time, temperature = read_filtrated_datafile(df,low,high)
    # polynomial fit
    z          = np.polyfit(time,conversion,pdeg)
    polynomial = np.poly1d(z)
    t_polfit   = np.linspace(time[0],time[-1],1000) # interpolate to these new points
    a_polfit   = polynomial(t_polfit)
    # get the reaction rate from the polynomial
    dadt = np.array([derivative(polynomial,ti,dx=1e-6) for ti in t_polfit])
    # interpolate the derivative
    f = interpolate.interp1d(t_polfit,dadt,fill_value='extrapolate')
    # calculate the derivative in the old time levels
    dadt = f(time)
    return dadt

def data2Exprate(Csv,low,high):
    # derivative of the actual experimental data
    # get dataframe
    df = pd.read_csv(Csv)
    # data
    conversion, time, temperature = read_filtrated_datafile(df,low,high)
    # numerical derivation of the experimental data
    der = np.diff( conversion ) / np.diff( time )
    # new time points
    t   = (time[:-1] + time[1:]) / 2.0
    # interpolate the derivative
    f = interpolate.interp1d(t,der,fill_value='extrapolate')
    # calculate the derivative in the old time levels
    dadt = f(time)
    return dadt

# simple scatter plot the experimental data
def export_experimental_reaction(DATA_DIR,OUTPUT_DIR,pdeg,npoints):

    # pdeg: degree of the polynomial
    # npoints: numper of polynomial interpolation points

    # make output directory
    DIR = os.path.join(OUTPUT_DIR,'experimental')
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # concentration range
    low  = 0.0
    high = 1.0
    
    # get the csv data in a list
    Csvs = get_data(DATA_DIR)
    # filnames
    fnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]

    for indx, Csv in enumerate(Csvs):
        
        # get experimental conversion
        df = pd.read_csv(Csv)
        conversion, time, temperature = read_filtrated_datafile(df,low,high)
        # experimental reaction rate from polynomial conversion
        dadt_polynomial = data2Polrate(Csv,low,high,pdeg,npoints)
        # experimental reaction rate from actual conversion
        dadt_numerical  = data2Exprate(Csv,low,high)

        data = {
            'time'            : time,
            'dadt_polynomial' : dadt_polynomial,
            'dadt_numerical'  : dadt_numerical,
            'temperature'     : temperature
        }

        rate_df       = pd.DataFrame(data)
        csv_name = fnames[indx] + '_reaction_rate.csv'
        rate_df.to_csv(os.path.join(DIR,csv_name),index=False)

