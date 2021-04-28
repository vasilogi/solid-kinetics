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
def sortOnData(bnames,l):
    # bnames : list of names on which data will be sorted
    # l      : list to sort
    # sl     : sorted list
    sl = []
    for bn in bnames:
        for csv in l:
            if bn in csv:
                sl.append(csv)
    return sl

def integralRegressionGraphs(DATA_DIR,OUTPUT_DIR,low,high,npoints):

    # get names (without format suffix) of the data csv files
    # bnames : base names
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]
    # paths of the experimental data csv files
    data_Csvs = get_data(DATA_DIR)
    # metrics directory
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'integral_regression')
    # paths of the metrics from the integral regression
    metrics_Csvs = get_data(METRICS_DIR)

    # zip data files and metrics
    data    = sortOnData(bnames,data_Csvs)
    metrics = sortOnData(bnames,metrics_Csvs)
    data_and_metrics = list(zip(data,metrics))
    
    # loop over all data
    for i_csv, csv in enumerate(data_and_metrics):

        # make directory for the graphs
        DIR       = os.path.join(METRICS_DIR,'png')
        GRAPH_DIR = os.path.join(DIR, bnames[i_csv])
        if not os.path.exists(GRAPH_DIR):
            os.makedirs(GRAPH_DIR)

        # data dataframe
        data_df    = pd.read_csv(csv[0])
        # metrics dataframe
        metrics_df = pd.read_csv(csv[1])
        # data
        conversion, time, temperature = read_filtrated_datafile(data_df,low,high)
        # read variable units
        timeUnits, massUnits, tempUnits = read_units(data_df)

        modelNames = metrics_df['model'].tolist()
        ks     = metrics_df['k_arrhenius'].to_numpy()

        # loop over models
        for i_model, modelName in enumerate(modelNames):
            # pick up a model
            model = Model(modelName)
            # choose the corresponding arrhenius rate constant
            k = ks[i_model]
            # calculate experimental integral reaction rate
            y = np.array( [ model.g(a) for a in conversion ] )
            x = time
            # fit
            tfit = np.linspace(time[0], time[-1], num=npoints)
            yfit = k*tfit
            xfit = tfit

            # export a graph for the fitting of the integral reaction rate
            fig  = plt.figure()
            fname = modelName + '_' + bnames[i_csv] + '_integral_regression.'+graph_format
            Plot = os.path.join(GRAPH_DIR,fname)
            plt.scatter(x, y, s=10, label='experimental')
            plt.plot(xfit, yfit, lw=lwidth, label=r'kt')
            plt.legend()
            plt.ylabel(r'g (a)')
            plt.xlabel('time [' + timeUnits +']')
            plt.tight_layout()
            plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
            plt.close() # to avoid memory warnings









npoints = 1000
integralRegressionGraphs(DATA,OUTPUT,low,high,npoints)























# bnames = [f.split('.csv')[0] for f in os.listdir(DATA)]
# data_Csvs = get_data(DATA)

# WDIR = os.path.join(OUTPUT, 'integral_regression')
# metrics_Csvs = get_data(WDIR)
# # filnames
# # fnames = [f.split('.csv')[0] for f in os.listdir(WDIR) if 'csv' in f]

# # zip data files and metrics
# data = []
# for bn in bnames:
#     for csv in data_Csvs:
#         if bn in csv:
#             data.append(csv)
            
# metrics = []
# for bn in bnames:
#     for csv in metrics_Csvs:
#         if bn in csv:
#             metrics.append(csv)

# data_and_metrics = list(zip(data,metrics))
            
# # for csv in data_and_metrics:
# csv = data_and_metrics[0]

# data_df    = pd.read_csv(csv[0])
# metrics_df = pd.read_csv(csv[1])

# # data
# conversion, time, temperature = read_filtrated_datafile(data_df,low,high)
# # read variable units
# timeUnits, massUnits, tempUnits = read_units(data_df)

# modelNames = metrics_df['model'].tolist()
# ks     = metrics_df['k_arrhenius'].to_numpy()

# modelName = modelNames[0]
# model = Model(modelName)
# k = ks[0]

# # data

# y = np.array( [ model.g(a) for a in conversion ] )
# x = time

# # fit

# npoints = 1000

# tfit = np.linspace(time[0], time[-1], num=1000)
# yfit = k*tfit
# xfit = tfit

# fig  = plt.figure()
# # export a graph for the fitting of the integral reaction rate

# DIR = os.path.join(WDIR, 'png')
# DIR = os.path.join(DIR, bnames[0])
# if not os.path.exists(DIR):
#     os.makedirs(DIR)

# fname = modelName + '_' + bnames[0] + '_integral_regression.'+graph_format
# Plot = os.path.join(DIR,fname)
# plt.scatter(x, y, s=10, label='experimental')
# plt.plot(xfit, yfit, lw=lwidth, label=r'kt')
# # plt.xlim(0,)
# # plt.ylim(0,1.0)
# plt.legend()
# plt.ylabel(r'g (a)')
# plt.xlabel('time [' + timeUnits +']')
# plt.tight_layout()
# plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
# # plt.close() # to avoid memory warnings