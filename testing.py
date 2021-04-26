import os
import numpy as np
import pandas as pd
from modules.file_handlers import read_filtrated_datafile, get_data, read_units
from modules.regressors import MSE
import matplotlib.pyplot as plt
from modules.reaction_models import Model
from modules.arrhenius import rateConstant, conv2mass
import random
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Local application imports
from modules.reaction_models import Model

modelNames = ["A2","A3","A4","D1","D2","D3","D4","F0","F1","F2","F3","P2","P3","P4","R2","R3"]

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

# limit conversion fraction
low  = 0.05
high = 0.95

from modules.integral_regression import data2integralFit

data2integralFit(DATA,OUTPUT,modelNames,low,high)

# Fit the conversion fraction with a polynomial
# get data files
Csvs = get_data(DATA)

Csv = Csvs[0]

# get dataframe
df = pd.read_csv(Csv)
# data
conversion, time, temperature = read_filtrated_datafile(df,low,high)

# fit
fit_degree = 9 # degree of the polynomial
z          = np.polyfit(time,conversion,fit_degree)
polynomial = np.poly1d(z) 
t_polfit   = np.linspace(time[0],time[-1],1000) # interpolate to these new points
a_polfit   = polynomial(t_polfit)

# plt.scatter(time,conversion,s=10)
# plt.plot(t_polfit, a_polfit)

model = Model(modelNames[5])
k = 0.0133114307850126

from scipy.misc import derivative

dadt = np.array([derivative(polynomial,ti,dx=1e-6) for ti in t_polfit])

plt.scatter(t_polfit,dadt,s=10)

der = np.diff( conversion ) / np.diff( time )
x2  = (time[:-1] + time[1:]) / 2

yfit = np.array([model.alpha(t, k) for t in t_polfit])
# plt.plot(t_polfit, yfit)
frate = np.array([k*model.f(c) for c in yfit])

plt.scatter(x2,der)

plt.plot(t_polfit, frate)