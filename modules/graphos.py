# Standard library imports
import os

# Third party imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Local application imports
from modules.arrhenius import mass2conv
from modules.file_handlers import read_filtrated_datafile, read_units, get_data
from modules.reaction_models import Model
from modules.reaction_rate_numerics import data2Polrate, data2Exprate
from modules.euclidean_distance import get_best_model
from modules.regressors import activation_enthalpy
from modules.arrhenius import csv2Arrhenius, rateConstant

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
    # filnames
    fnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]

    fig  = plt.figure()
    # export a graph for the fitting of the integral reaction rate
    Plot = os.path.join(DIR,'experimental_conversion.'+graph_format)
    
    for indx, Csv in enumerate(Csvs):
        df = pd.read_csv(Csv)
        # read data file
        conversion, time, temperature   = read_filtrated_datafile(df,low,high)
        # read variable units
        timeUnits, tempUnits = read_units(df)
        plt.scatter(time,conversion,s=10,label=str(temperature)+tempUnits)
        # export experimental data
        data = {
            'time'        : time,
            'conversion'  : conversion,
            'temperature' : temperature,
            'temperature_units': tempUnits
        }
        df = pd.DataFrame(data)
        csv_name = fnames[indx] + '_experimental_conversion.csv'
        df.to_csv(os.path.join(DIR,csv_name),index=False)
    
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
    criteria.remove('temperature')
    criteria.remove('temperature_units')
    if 'k_arrhenius' in criteria:
        criteria.remove('k_arrhenius')
    

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

    # sns.set_theme()
    
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
        timeUnits, tempUnits = read_units(data_df)

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

def conversionRegressionGraphs(DATA_DIR,OUTPUT_DIR,low,high,npoints):

    # get names (without format suffix) of the data csv files
    # bnames : base names
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]
    # paths of the experimental data csv files
    data_Csvs = get_data(DATA_DIR)
    # metrics directory
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'conversion_regression')
    # paths of the metrics from the conversion regression
    metrics_Csvs = get_data(METRICS_DIR)
    # filter proper csvs
    metrics_Csvs = [f for f in metrics_Csvs if 'conversion_regression_accuracy' in f]

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
        timeUnits, tempUnits = read_units(data_df)

        modelNames = metrics_df['model'].tolist()
        ks     = metrics_df['k_arrhenius'].to_numpy()

        # loop over models
        for i_model, modelName in enumerate(modelNames):
            # pick up a model
            model = Model(modelName)
            # choose the corresponding arrhenius rate constant
            k = ks[i_model]
            # calculate experimental conversion
            y = conversion
            x = time
            if modelName not in ['D2','D4']:
                # fit
                tfit = np.linspace(time[0], time[-1], num=npoints)
                yfit = np.array( [ model.alpha(t, k) for t in tfit ] )
                xfit = tfit

                # export a graph for the fitting of the integral reaction rate
                fig  = plt.figure()
                fname = modelName + '_' + bnames[i_csv] + '_conversion_regression.'+graph_format
                Plot = os.path.join(GRAPH_DIR,fname)
                plt.scatter(x, y, s=10, label='experimental')
                plt.plot(xfit, yfit, lw=lwidth, label=modelName)
                plt.legend()
                plt.ylabel(r'conversion')
                plt.xlabel('time [' + timeUnits +']')
                plt.ylim(0.0, 1.0)
                plt.tight_layout()
                plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
                plt.close() # to avoid memory warnings

def differentialRegressionGraphs(DATA_DIR,OUTPUT_DIR,low,high,npoints):

    # get names (without format suffix) of the data csv files
    # bnames : base names
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]
    # paths of the experimental data csv files
    data_Csvs = get_data(DATA_DIR)
    # metrics directory
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'differential_regression')
    # paths of the metrics from the conversion regression
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
        timeUnits, tempUnits = read_units(data_df)

        modelNames = metrics_df['model'].tolist()
        ks     = metrics_df['k_arrhenius'].to_numpy()

        # loop over models
        for i_model, modelName in enumerate(modelNames):
            # pick up a model
            model = Model(modelName)
            # choose the corresponding arrhenius rate constant
            k = ks[i_model]
            # calculate experimental conversion
            y = conversion
            x = time
            if modelName not in ['D2','D4']:
                # fit
                tfit = np.linspace(time[0], time[-1], num=npoints)
                yfit = np.array( [ model.alpha(t, k) for t in tfit ] )
                xfit = tfit

                # export a graph for the fitting of the integral reaction rate
                fig  = plt.figure()
                fname = modelName + '_' + bnames[i_csv] + '_differential_regression.'+graph_format
                Plot = os.path.join(GRAPH_DIR,fname)
                plt.scatter(x, y, s=10, label='experimental')
                plt.plot(xfit, yfit, lw=lwidth, label=modelName)
                plt.legend()
                plt.ylabel(r'conversion')
                plt.xlabel('time [' + timeUnits +']')
                plt.ylim(0.0, 1.0)
                plt.tight_layout()
                plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
                plt.close() # to avoid memory warnings

def rateFitGraphs(DATA_DIR,OUTPUT_DIR,low,high,pdeg,npoints,fitExp):

    # get names (without format suffix) of the data csv files
    # bnames : base names
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]
    # paths of the experimental data csv files
    data_Csvs = get_data(DATA_DIR)
    # metrics directory
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'conversion_regression')
    # paths of the metrics from the conversion regression
    metrics_Csvs = get_data(METRICS_DIR)
    # filter proper csvs
    metrics_Csvs = [f for f in metrics_Csvs if 'conversion_regression_accuracy' in f]

    # zip data files and metrics
    data    = sortOnData(bnames,data_Csvs)
    metrics = sortOnData(bnames,metrics_Csvs)
    data_and_metrics = list(zip(data,metrics))
    
    # loop over all data
    for i_csv, csv in enumerate(data_and_metrics):

        # make directory for the graphs
        DIR       = os.path.join(METRICS_DIR,'png')
        DIR       = os.path.join(DIR,'rate_fit')
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
        timeUnits, tempUnits = read_units(data_df)

        # experimental reaction rate from polynomial conversion
        dadt_polynomial = data2Polrate(csv[0],low,high,pdeg,npoints)
        # experimental reaction rate from actual conversion
        dadt_numerical  = data2Exprate(csv[0],low,high)

        modelNames = metrics_df['model'].tolist()
        ks         = metrics_df['k_arrhenius'].to_numpy()

        # calculate experimental reaction rate
        if fitExp:
            y = dadt_numerical
        else:
            y = dadt_polynomial

        x = time

        # loop over models
        for i_model, modelName in enumerate(modelNames):
            # pick up a model
            model = Model(modelName)
            # choose the corresponding arrhenius rate constant
            k = ks[i_model]

            if modelName not in ['D2','D4']:
                # calculate the modeled differential reaction rate
                tfit = np.linspace(time[0], time[-1], num=npoints)
                yfit = np.array( [ model.alpha(t, k) for t in tfit ] )
                dadt_model = np.array( [k*model.f(a) for a in yfit] )
                yfit = dadt_model
                xfit = tfit
                # export a graph for the fitting of the integral reaction rate
                fig  = plt.figure()
                if fitExp:
                    ext = '_experimental_rate_fit.'
                else:
                    ext = '_polynomial_rate_fit.'
                fname = modelName + '_' + bnames[i_csv] + ext + graph_format
                Plot = os.path.join(GRAPH_DIR,fname)
                plt.scatter(x, y, s=10, label='experimental')
                plt.plot(xfit, yfit, lw=lwidth, label=modelName)
                plt.legend()
                plt.ylabel(r'reaction rate')
                plt.xlabel('time [' + timeUnits +']')
                plt.tight_layout()
                plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
                plt.close() # to avoid memory warnings

def export_kinetic_triplet(OUTPUT_DIR):
    GRAPH_DIR = os.path.join(OUTPUT_DIR, 'desicion')
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)
    
    best_model = get_best_model(OUTPUT_DIR)

    WDIR = os.path.join(OUTPUT_DIR,'conversion_regression')

    Csvs = [os.path.join(WDIR,i) for i in os.listdir(WDIR) if 'conversion_regression_accuracy.csv' in i]

    k, T = csv2Arrhenius(Csvs,best_model)

    fit_data = activation_enthalpy(k,T)

    df = pd.DataFrame.from_dict(fit_data,orient='index')
    df.to_csv(os.path.join(GRAPH_DIR,'kinetic_triplet.csv'))

    x_fit = np.linspace(round(min(T)-10),round(max(T)+10),500)
    k_fit = np.array( [ rateConstant(fit_data['frequency_factor'],fit_data['activation_enthalpy'],T) for T in x_fit ] )

    fig  = plt.figure()
    fname = 'best_model_prediction.' + graph_format

    Plot = os.path.join(GRAPH_DIR,fname)
    plt.scatter(T, k, s=50, label='experimental')
    plt.plot(x_fit, k_fit, lw=lwidth, label='prediction')
    plt.legend()
    plt.xlim(min(x_fit),max(x_fit))
    plt.ylabel('Arrhenius rate constant')
    plt.xlabel('temperature (K)')
    plt.tight_layout()
    plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
    plt.close() # to avoid memory warnings

def graph_isoconversional_enthalpy(OUTPUT_DIR):
    WDIR = os.path.join(OUTPUT_DIR,'isoconversional')
    df = pd.read_csv(os.path.join(WDIR,'isoconversional_energy.csv'))
    fig  = plt.figure()
    fname = 'isoconversional_enthalpy.' + graph_format

    Plot = os.path.join(WDIR,fname)

    fig = plt.figure()
    plt.scatter(df['conversion'], df['activation_enthalpy'], s=50)
    plt.ylabel('activation enthalpy [kJ/mol]')
    plt.xlabel('conversion')
    plt.xlim(0.0,1.0)
    plt.tight_layout()
    plt.savefig(Plot, format=graph_format, dpi=graph_dpi)
    plt.close() # to avoid memory warnings
    