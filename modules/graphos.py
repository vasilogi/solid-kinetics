# Standard library imports
import os

# Third party imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Local application imports
from modules.arrhenius import mass2conv
from modules.file_handlers import read_filtrated_datafile, read_units, get_data

# basic plot settings
graph_format = 'png'
graph_dpi    = 300
font_size    = 13
lwidth       = 3

# simple scatter plot the experimental data
def graph_experimental_data(DATA_DIR,OUTPUT_DIR):

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
    fnames = os.listdir(DATA_DIR)

    # create DIR directory
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    fig  = plt.figure()
    # export a graph for the fitting of the integral reaction rate
    Plot = os.path.join(DIR,'experimental_conversion.'+graph_format)
    
    for indx, Csv in enumerate(Csvs):
        df = pd.read_csv(Csv)
        # read data file
        conversion, time, temperature   = read_filtrated_datafile(df,low,high)
        # read variable units
        timeUnits, massUnits, tempUnits = read_units(df)
        plt.scatter(time,conversion,s=10,label=str(temperature)+tempUnits)
        # export experimental data
        data = {
            'time': time,
            'conversion': conversion
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(DIR,fnames[indx]),index=False)
    
    plt.xlim(0,)
    plt.ylim(0,1.0)
    plt.legend()
    plt.ylabel('conversion')
    plt.xlabel('time [' + timeUnits +']')
    plt.tight_layout()
    plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
    plt.close() # to avoid memory warnings

# heatmap the accuracy measures
# function for one dataframe and fname
def df2heatmap(WDIR,df,fname):

    # rank for each measure
    def df2rank(dframe,measure):
        # dframe  : the accuracy data dataframe
        # measure : particular measure (column name)
        # returns a ranking list

        # sort dataframe by the specific measure
        if measure == 'rSquared':
            df = dframe.sort_values(by=[measure],ascending=False)
        else:
            df = dframe.sort_values(by=[measure])
        # get only the model and specific measure columns into a new frame
        selDF = df[['model',measure]]
        # reset the index of this frame so the best one starts from 0
        selDF.reset_index(inplace=True)
        # sort the dataframe by the initial index, e.g. by the initial models column
        # the real index of the dataframe (not the column named index) is the ranking
        # of the models
        selDF = selDF.sort_values(by=['index'])
        rank  = [i+1 for i in selDF.index.to_list()]

        return rank

    # get only the numerical columns regarding accuracy measures
    criteria = df.columns.to_list()
    criteria.remove('model')
    criteria.remove('k_arrhenius')
    criteria.remove('temperature')

    # get the models in a list
    models = df['model'].tolist()

    # initialise ranking dictionary
    ranking = {'models': models}

    # loop over all criteria
    for measure in criteria:
        # get the ranking for the particular measure
        rank = df2rank(df, measure)
        # parse it to dictionary
        ranking[measure] = rank

    sns.set_theme()
    
    # create ranking dataframe
    rankDF = pd.DataFrame(ranking)
    rankDF = rankDF.set_index('models')
    # heatmap plot
    Plot = os.path.join(WDIR,fname+'.'+graph_format)
    fig    = plt.figure()
    sns.heatmap(rankDF, vmin=1, vmax=len(models), cmap='RdBu', linecolor='black', linewidths=0.5, annot=True)
    plt.tight_layout()
    plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
    plt.close()

# export heatmaps for all csvs for all directories
def measures2heatmaps(OUTPUT_DIR):
    ext = ['integral_regression', 'differential_regression', 'conversion_regression']
    # loop over directories
    for DIR in ext:
        # set working directory
        WDIR = os.path.join(OUTPUT_DIR, DIR)
        # get all csvs in the directory
        Csvs = get_data(WDIR)
        # filnames
        fnames = [f.split('.csv')[0] for f in os.listdir(WDIR) if 'csv' in f]
        # loop over the csv files in this DIR
        for indx, Csv in enumerate(Csvs):
            df    = pd.read_csv(Csv)
            fname = fnames[indx]
            df2heatmap(WDIR, df, fname)