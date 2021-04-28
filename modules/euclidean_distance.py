# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.file_handlers import get_data

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
    bnames = [f.split('.csv')[0] for f in os.listdir(DATA_DIR)]
    
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