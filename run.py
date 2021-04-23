# Standard library imports
import os

# Local application imports
from modules.graphos import graph_experimental_data

# DIRECTORIES
MAIN_DIR  = os.getcwd()                     # current working directory
DATA      = os.path.join(MAIN_DIR,'data')   # data directory
OUTPUT    = os.path.join(MAIN_DIR,'output') # output directory

# limit conversion fraction
low  = 0.05
high = 0.95

for 

    # load the particular csv as chosen by the files-dropdown
    case = filename

    # read a data file
    conversion, time, temperature = read_filtrated_datafile(case,low,high)

    # perform non-linear regression and return the fitting information
    df = comprehensiveRegressor(time, conversion, modelNames)

    # calculate the convergence criterion
    data = convergenceData(df)