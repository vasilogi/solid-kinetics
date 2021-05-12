import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.euclidean_distance import get_best_model
from modules.regressors import activation_enthalpy
from modules.arrhenius import csv2Arrhenius, rateConstant
from modules.file_handlers import get_data
from modules.file_handlers import read_filtrated_datafile
from scipy.interpolate import interp1d

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def integral_isoconversional(DATA_DIR,OUTPUT_DIR,low,high):

    def interpolate_time(y,x,interpolated_x):
        # arguments
        # x (numpy array) : conversion
        # y (numpy array) : time
        
        # returns
        # the time for the interpolated points
        
        y               = time
        x               = conversion
        interpol        = interp1d(x,y,kind='nearest',fill_value="extrapolate")
        return interpol(interpolated_x)

    def isoconversional_enthalpy(time,temperature):
        # arguments 
        # time (numpy array)
        # temperature (numpy array)

        # returns
        # the activation enthalpy (Ea) in a list like [activation enthalpy, mean square error]
        # and the ln[g(a)A] factor

        # gas constant 
        R = 8.31446261815324 # J K-1 mol-1

        x = 1.0/temperature
        y = np.log(time)

        x = x.reshape((-1, 1))

        # linear regression for the logarithmic Arrhenius equation
        regr   = LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)

        Ea  = regr.coef_[0]*R*1.0e-3 # in kJ mol-1
        gA  = regr.intercept_
        MSE = mean_squared_error(y, y_pred)
        R2   = r2_score(y_pred , y)

        return [Ea,MSE], gA

    Csvs = get_data(DATA_DIR)

    npoints = 10
    interpolated_conversion = np.linspace(low,high,npoints)

    isoconversional_data = {'conversion': interpolated_conversion}

    for csv in Csvs:
        df = pd.read_csv(csv)
        # read a data file
        conversion, time, temperature   = read_filtrated_datafile(df,low,high)
        if df['temperature units'][0] == 'C':
            theta = df['temperature'].to_numpy()
            T = Celsius2Kelvin(theta)
        else:
            T = df['temperature'].to_numpy()
        temperature = T[0]
        # get time in the specified interpolated_conversion points
        interpolated_time = interpolate_time(time,conversion,interpolated_conversion)
        isoconversional_data.update({str(temperature):interpolated_time})

    df = pd.DataFrame.from_dict(isoconversional_data)

    # linear regression

    y = df['conversion'].to_numpy()

    temperature = [float(i) for i in df.columns.values if i != 'conversion']
    temperature = np.array(temperature)

    Ea        = [] # Activation energy (kJ/mol)
    intercept = [] # Intercept ln[A g(a)]
    MSE       = [] # Standard deviation
    R2        = [] # Determination coefficient

    dfSize = df.shape[0]
    for i in range(dfSize):
        time = df.iloc[i,1::].to_numpy()
        enthalpy, gA = isoconversional_enthalpy(time,temperature)
        Ea.append(enthalpy[0])
        MSE.append(enthalpy[1])
        intercept.append(gA)
        
    isoconversional_data = {'activation_enthalpy': Ea, 'std': MSE, 'intercept': intercept, 'conversion': y}
    df = pd.DataFrame.from_dict(isoconversional_data)

    # make output directory
    DIR = os.path.join(OUTPUT_DIR,'isoconversional')
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    df.to_csv(os.path.join(DIR,'isoconversional_energy.csv'),index=False)

    pass