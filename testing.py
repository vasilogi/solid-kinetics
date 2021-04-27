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

graph_format = 'png'
graph_dpi    = 300
font_size    = 13
lwidth       = 3

# get conversion regression data

WDIR = os.path.join(OUTPUT, 'integral_regression')
Csvs = get_data(WDIR)
# filnames
fnames = [f.split('.csv')[0] for f in os.listdir(WDIR) if 'csv' in f]

indx = 2
Csv = Csvs[indx]
df = pd.read_csv(Csv)
fname = fnames[indx]


from modules.integral_regression import data2integralFit

data2integralFit(DATA,OUTPUT,modelNames,low,high)

# Fit the conversion fraction with a polynomial
# get data files
Csvs = get_data(DATA)

Csv = Csvs[0]

df = pd.read_csv(Csv)
conversion, time, temperature = read_filtrated_datafile(df,low,high)

model = Model(modelNames[7])
k = 0.0133114307850126


from modules.reaction_rate_numerics import data2Polrate, data2Exprate

dadt = data2Polrate(Csv,low,high,9,1000)
plt.scatter(time,dadt,s=10)


# yfit = np.array([model.alpha(t, k) for t in t_polfit])
# # plt.plot(t_polfit, yfit)
frate = np.array([k*model.f(c) for c in conversion])
dadt = data2Exprate(Csv,low,high)
plt.scatter(time,dadt)

plt.plot(time, frate)

from modules.gof import resREr

y = dadt
yfit = frate
s = resREr(y, yfit)
print(s)