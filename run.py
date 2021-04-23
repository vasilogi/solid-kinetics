# Standard library imports
import os

# Third party imports
import pandas as pd

# Local application imports
from modules.graphos import graph_experimental_data
from modules.file_handlers import read_filtrated_datafile, get_data
from modules.regressors import comprehensiveRegressor
from modules.euclidean_distance import convergenceData

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
# get corresponding file names
filenames = [f.split('.csv')[0] for f in os.listdir(DATA) if 'csv' in f]

# plot solely the experimental data
graph_experimental_data(DATA,OUTPUT)

for indx, Csv in enumerate(Csvs):
    # filename
    fname = filenames[indx]
    # read data
    df = pd.read_csv(Csv)
    conversion, time, temperature = read_filtrated_datafile(df,low,high)
    # perform non-linear regression and return the fitting information
    df = comprehensiveRegressor(time, conversion, modelNames)
    # save regression data
    df.to_csv(os.path.join(OUTPUT,fname+'_regression_data.csv'),index=False)
    # calculate the convergence criterion
    convergence_data = convergenceData(df)
    # save convergence criteria data
    convergence_data.to_csv(os.path.join(OUTPUT,fname+'_convergence_criteria_data.csv'),index=False)