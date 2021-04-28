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



# def sortOnData(bnames,l):
#     # bnames : list of names on which data will be sorted
#     # l      : list to sort
#     # sl     : sorted list
#     sl = []
#     for bn in bnames:
#         for csv in l:
#             if bn in csv:
#                 sl.append(csv)
#     return sl

def criteria2desicionIndex(DATA_DIR,OUTPUT_DIR,measure,fitExp):
    def findMeasure(df,measure,modelname):
        # return the value for a particular measure and model
        return df[ df['model'] == modelname ][measure].iloc[0]

    def df2erarray(df,measure,models):
        # return all the measure values for all models
        err = np.zeros(len(models))
        for i_model, modelname in enumerate(models):
            err[i_model] = findMeasure(df, measure, modelname)
        return err

    # output directory for desicions
    ODIR = os.path.join(OUTPUT_DIR,'desicion')
    if not os.path.exists(ODIR):
        os.makedirs(ODIR)

    # get names of the experimental data
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA)]
    
    # loop over files (temperatures)
    for bn in bnames:

        # conversion regression directory
        DIR = os.path.join(OUTPUT_DIR,'conversion_regression')
        Csvs = get_data(DIR)

        convReg_csvs    = [f for f in Csvs if 'conversion_regression_accuracy' in f]
        exprateFit_csvs = [f for f in Csvs if 'experimental_rate_fit_accuracy' in f]
        polrateFit_csvs = [f for f in Csvs if 'polynomial_rate_fit_accuracy' in f]

        # paths of the csv files
        convReg_csv    = [i for i in convReg_csvs if bn in i][0]
        exprateFit_csv = [i for i in exprateFit_csvs if bn in i][0]
        polrateFit_csv = [i for i in polrateFit_csvs if bn in i][0]

        # integral regression directory
        DIR = os.path.join(OUTPUT_DIR,'integral_regression')
        Csvs = get_data(DIR)
        interateReg_csvs = [f for f in Csvs if 'integral_regression_accuracy' in f]
        # path of the csv file
        interateReg_csv    = [i for i in interateReg_csvs if bn in i][0]

        # differential regression directory
        DIR  = os.path.join(OUTPUT_DIR,'differential_regression')
        Csvs = get_data(DIR)
        diffrateReg_csvs = [f for f in Csvs if 'differential_regression_accuracy' in f]
        # path of the csv file
        diffrateReg_csv = [i for i in diffrateReg_csvs if bn in i][0]

        # get the modelnames
        models = pd.read_csv(convReg_csv)['model'].tolist()

        # get corresponding dataframes
        df = pd.read_csv( convReg_csv )
        convReg_err = df2erarray(df,measure,models)
        if fitExp:
            df = pd.read_csv( exprateFit_csv )
            rateFit_err = df2erarray(df,measure,models)
        else:
            df = pd.read_csv( polrateFit_csv )
            rateFit_err = df2erarray(df,measure,models)
        df = pd.read_csv( interateReg_csv )
        interateReg_err = df2erarray(df,measure,models)
        df = pd.read_csv( diffrateReg_csv )
        diffrateReg_err = df2erarray(df,measure,models)

        # error data
        error_data = tuple(zip(convReg_err,rateFit_err,interateReg_err,diffrateReg_err))

        # euclidean norm
        a = np.array(error_data[0])
        b = np.zeros(len(a))        # base vector (ideal values)

        # calculate the L2_norm of the error vector for each model
        L2_norm = np.zeros(len(models))
        for i_model, modelname in enumerate(models):
            a = np.array(error_data[i_model])
            L2_norm[i_model] = np.linalg.norm(a-b)
        
        # get error fitting data
        measure_key = measure+'_L2_norm'
        data = { 'model': models, measure_key: L2_norm }
        des_df = pd.DataFrame(data)
        des_df.sort_values(by=[measure_key], inplace=True)
        
        # export csv
        Expname = os.path.join( ODIR, bn + '_' + measure + '_desicion.csv' )
        des_df.to_csv(Expname,index=False)




measure = 'resREr'
fitExp = True
criteria2desicionIndex(DATA,OUTPUT,measure,fitExp)













# bnames = [f.split('.csv')[0] for f in os.listdir(DATA)]

# bn = bnames[0]

# DIR = os.path.join(OUTPUT,'conversion_regression')
# Csvs = get_data(DIR)

# convReg_csvs    = [f for f in Csvs if 'conversion_regression_accuracy' in f]
# exprateFit_csvs = [f for f in Csvs if 'experimental_rate_fit_accuracy' in f]
# polrateFit_csvs = [f for f in Csvs if 'polynomial_rate_fit_accuracy' in f]

# convReg_csv    = [i for i in convReg_csvs if bn in i][0]
# exprateFit_csv = [i for i in exprateFit_csvs if bn in i][0]
# polrateFit_csv = [i for i in polrateFit_csvs if bn in i][0]

# DIR = os.path.join(OUTPUT,'integral_regression')
# Csvs = get_data(DIR)
# interateReg_csvs = [f for f in Csvs if 'integral_regression_accuracy' in f]
# interateReg_csv    = [i for i in interateReg_csvs if bn in i][0]

# DIR  = os.path.join(OUTPUT,'differential_regression')
# Csvs = get_data(DIR)
# diffrateReg_csvs = [f for f in Csvs if 'differential_regression_accuracy' in f]
# diffrateReg_csv = [i for i in diffrateReg_csvs if bn in i][0]



# models = pd.read_csv(convReg_csv)['model'].tolist()
# measure = 'resREr'
# def findMeasure(df,measure,modelname):
#     return df[ df['model'] == modelname ][measure].iloc[0]

# def df2erarray(df,measure,models):
#     err = np.zeros(len(models))
#     for i_model, modelname in enumerate(models):
#         err[i_model] = findMeasure(df, measure, modelname)
#     return err


# df = pd.read_csv( convReg_csv )
# convReg_err = df2erarray(df,measure,models)
# df = pd.read_csv( polrateFit_csv )
# polrateFit_err = df2erarray(df,measure,models)
# df = pd.read_csv( interateReg_csv )
# interateReg_err = df2erarray(df,measure,models)
# df = pd.read_csv( diffrateReg_csv )
# diffrateReg_err = df2erarray(df,measure,models)

# error_data = tuple(zip(convReg_err,polrateFit_err,interateReg_err,diffrateReg_err))

# # euclidean norm
# a = np.array(error_data[0])
# b = np.zeros(len(a))

# L2_norm = np.zeros(len(models))
# for i_model, modelname in enumerate(models):
#     a = np.array(error_data[i_model])
#     L2_norm[i_model] = np.linalg.norm(a-b) 

# measure = measure+'_L2_norm'
# data = { 'model': models, measure: L2_norm }

# df = pd.DataFrame(data)

# df.sort_values(by=[measure], inplace=True)
    
# # convReg_csvs     = sortOnData(bnames,convReg_csvs)
# # exprateFit_csvs  = sortOnData(bnames,exprateFit_csvs)
# # polrateFit_csvs  = sortOnData(bnames,polrateFit_csvs)
# # interateReg_csvs = sortOnData(bnames,interateReg_csvs)
# # diffrateReg_csvs = sortOnData(bnames,diffrateReg_csvs)

# # models = pd.read_csv(convReg_csvs[0])['model'].tolist()

# # for i_csv in range(len(convReg_csvs)):
# #     error_1 = pd.read_csv(convReg_csvs[i_csv])['resREr'].to_numpy()
# #     error_2 = pd.read_csv(exprateFit_csvs[i_csv])['resREr'].to_numpy()
# #     error_3 = pd.read_csv(polrateFit_csvs[i_csv])['resREr'].to_numpy()
# #     error_4 = pd.read_csv(interateReg_csvs[i_csv])['resREr'].to_numpy()
# #     error_5 = pd.read_csv(diffrateReg_csvs[i_csv])['resREr'].to_numpy()
    
# #     erVec = np.array( [] )
# #     for i_error in range(len(models)):
        

# # # filter the negative determination coefficient values
# # numDf = df._get_numeric_data()
# # numDf[numDf < 0] = 0

# # # calculate the Euclidean distance (fitting error)
# # p = numDf['MSE - alpha'].to_numpy()
# # q = numDf['MSE - integral'].to_numpy()
# # # r = numDf['MSE - differential'].to_numpy()
# # # dist = d(p,q,r)
# # dist = d(p,q)
# # data = {'model': df['model'].to_list(), 'fitting_error': dist}
# # return pd.DataFrame(data)























# # # bnames = [f.split('.csv')[0] for f in os.listdir(DATA)]
# # # data_Csvs = get_data(DATA)

# # # WDIR = os.path.join(OUTPUT, 'integral_regression')
# # # metrics_Csvs = get_data(WDIR)
# # # # filnames
# # # # fnames = [f.split('.csv')[0] for f in os.listdir(WDIR) if 'csv' in f]

# # # # zip data files and metrics
# # # data = []
# # # for bn in bnames:
# #     for csv in data_Csvs:
# #         if bn in csv:
# #             data.append(csv)
            
# # metrics = []
# # for bn in bnames:
# #     for csv in metrics_Csvs:
# #         if bn in csv:
# #             metrics.append(csv)

# # data_and_metrics = list(zip(data,metrics))
            
# # # for csv in data_and_metrics:
# # csv = data_and_metrics[0]

# # data_df    = pd.read_csv(csv[0])
# # metrics_df = pd.read_csv(csv[1])

# # # data
# # conversion, time, temperature = read_filtrated_datafile(data_df,low,high)
# # # read variable units
# # timeUnits, tempUnits = read_units(data_df)

# # modelNames = metrics_df['model'].tolist()
# # ks     = metrics_df['k_arrhenius'].to_numpy()

# # modelName = modelNames[0]
# # model = Model(modelName)
# # k = ks[0]

# # # data

# # y = np.array( [ model.g(a) for a in conversion ] )
# # x = time

# # # fit

# # npoints = 1000

# # tfit = np.linspace(time[0], time[-1], num=1000)
# # yfit = k*tfit
# # xfit = tfit

# # fig  = plt.figure()
# # # export a graph for the fitting of the integral reaction rate

# # DIR = os.path.join(WDIR, 'png')
# # DIR = os.path.join(DIR, bnames[0])
# # if not os.path.exists(DIR):
# #     os.makedirs(DIR)

# # fname = modelName + '_' + bnames[0] + '_integral_regression.'+graph_format
# # Plot = os.path.join(DIR,fname)
# # plt.scatter(x, y, s=10, label='experimental')
# # plt.plot(xfit, yfit, lw=lwidth, label=r'kt')
# # # plt.xlim(0,)
# # # plt.ylim(0,1.0)
# # plt.legend()
# # plt.ylabel(r'g (a)')
# # plt.xlabel('time [' + timeUnits +']')
# # plt.tight_layout()
# # plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
# # # plt.close() # to avoid memory warnings