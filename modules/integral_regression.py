# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
from modules.file_handlers import read_filtrated_datafile, get_data, read_units
from modules.regressors import integralRateRegression
from modules.gof import ssRes, MSE, resAEr, resREr, rSquared
from modules.reaction_models import Model


def data2integralFit(DATA_DIR,OUTPUT_DIR,modelNames,low,high):
    # low        : lower limit for conversion fraction
    # high       : upper limit for conversion fraction
    # DATA_DIR   : directory containing data
    # OUTPUT_DIR : output directory

    # make output directory
    DIR = os.path.join(OUTPUT_DIR,'integral_regression')
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # get csvs
    Csvs = get_data(DATA_DIR)

    # filnames
    fnames = os.listdir(DATA_DIR)

    for indx, Csv in enumerate(Csvs):
        # get dataframe
        df = pd.read_csv(Csv)
        # data
        conversion, time, temperature = read_filtrated_datafile(df,low,high)
        # read variable units
        timeUnits, massUnits, tempUnits = read_units(df)
        # accuracy criteria
        ss_res      = [] # sum of square residuals (ideal = 0)
        mse         = [] # mean square error (ideal = 0)
        res_AEr     = [] # residuals absolute error (ideal = 0)
        res_REr     = [] # residuals relative error (ideal = 0)
        r_Squared   = [] # determination coefficient (ideal = 1)
        k_arrhenius = [] # Arrhenius rate constant

        # loop over all models
        for modelName in modelNames:
            # pick up a model
            model = Model(modelName)
            # experimental integral reaction rate
            y = np.array([model.g(c) for c in conversion])
            # perform regression
            k, yfit = integralRateRegression(time,conversion, modelName)
            # calculate validation errors
            ss_res.append(ssRes(y,yfit))
            mse.append(MSE(y,yfit))
            res_AEr.append(resAEr(y,yfit))
            res_REr.append(resREr(y,yfit))
            r_Squared.append(rSquared(y,yfit))
            k_arrhenius.append(k)
        
        # export regression accuracy data
        error_data = {
            'model'       : modelNames,
            'ss_res'      : ss_res,
            'mse'         : mse,
            'resAEr'      : res_AEr,
            'resREr'      : res_REr,
            'rSquared'    : r_Squared,
            'k_arrhenius' : k_arrhenius
        }
        df = pd.DataFrame(error_data)
        prefix = fnames[indx].split('.csv')[0]
        df.to_csv(os.path.join(DIR,prefix + '_integral_regression_accuracy.csv'),index=False)