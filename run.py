# Standard library imports
import os

# Third party imports
# import pandas as pd

# Local application imports
from modules.graphos import graph_experimental_data
from modules.file_handlers import get_data
from modules.integral_regression import data2integralFit
from modules.conversion_regression import data2conversionFit
from modules.differential_regression import  data2differentialFit
from modules.graphos import measures2heatmaps

# models names supported in this software
modelNames = ["A2","A3","A4","D1","D2","D3","D4","F0","F1","F2","F3","P2","P3","P4","R2","R3"]

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

# limit conversion fraction
low  = 0.05
high = 0.95

# get data files
Csvs = get_data(DATA)

# plot and export solely the experimental data
graph_experimental_data(DATA,OUTPUT)

# perform linear regression on the integral rate experimental data
data2integralFit(DATA,OUTPUT,modelNames,low,high)

# perform non-linear regression on the exact conversion
data2conversionFit(DATA,OUTPUT,modelNames,low,high)

# perform non-linear regression on the differential rate experimental data
data2differentialFit(DATA,OUTPUT,modelNames,low,high)

# heatmap accuracy data
measures2heatmaps(OUTPUT)