from util.logger import logger
from util.date_utils import iter_months
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os
from prophet import Prophet
from config.app_config import config
import pickle
import argparse
import sys
import time
import string


PROPHET_MODEL = 'prophet'
feature_cols = ['ds','holiday','is_peak','code','temperature']
MONTH_NUMBER = 6


class ProphetService:

    @staticmethod
    def train_all_prophet():
        modle_type = {"load","meter"}
        model_dir = config.get_model_dir()
        data_dir = config.get_data_dir()
        logger.info(f"data_dir dir is {data_dir}")
        
        for mode_type in modle_type:
            ProphetService.train_prophet(data_dir,model_dir,mode_type,MONTH_NUMBER)




    @staticmethod
    def train_prophet(data_dir:string,model_dir:string,model_type:string,monthNumber:int):
        companys =    ProphetService.list_company(data_dir)
        
        for company in companys:
            loadpf = ProphetService.load_train_data(data_dir,company,monthNumber)    
            loadpf = loadpf.rename(columns={'time': 'ds', model_type: 'y'})            
            model = Prophet(changepoint_prior_scale=0.5,growth='flat', weekly_seasonality=True, daily_seasonality=True )
            
            model.add_regressor(feature_cols[1])
            model.add_regressor(feature_cols[2])
            model.add_regressor(feature_cols[3])
            model.add_regressor(feature_cols[4])
         
            model.fit(loadpf)           
            model_filename = ProphetService.get_model_fullpath(model_dir,company,model_type)
            
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def list_company(path): 
        subdirs = []    
        for item in os.listdir(path):
            subdirs.append(item)
        return    subdirs



    @staticmethod
    def get_data_fullpath(data_dir:string, company:string,year:int,month:int):
        filename = "data"+"-"+company+"-"+str(year)+"-"+f"{month:02d}"+".csv"    
        return os.path.join(data_dir,company,"data",filename)

    @staticmethod 
    def get_model_fullpath(model_dir:string, company:string,model_type:string):
        filename = PROPHET_MODEL+"-"+model_type+"-"+company+".pkl"      
        modelPath = os.path.join(model_dir,company,PROPHET_MODEL)
  
        if os.path.exists(modelPath) == False :
            os.makedirs(modelPath, exist_ok=True)  
        return os.path.join(modelPath,filename)

    @staticmethod
    def load_train_data(data_dir:string, company:string,num:int):
    
        dataframe = pd.DataFrame()
        now = datetime.now()
        current_date =    now.day
        current_month = now.month
        current_year = now.year
    
        for number in range(0,num):    
                
            new_month = ProphetService.get_last_n_month(current_year,current_month,number)
            new_year = ProphetService.get_last_year(current_year,current_month,number)    
            filename = ProphetService.get_data_fullpath(data_dir,company,new_year,new_month)
        
            if os.path.isfile(filename) == True:
                df_temp    = pd.read_csv(filename,sep=",")    
                dataframe =pd.concat( [dataframe,df_temp],ignore_index=True)
     
        dataframe = dataframe.reset_index()
        return dataframe    
    
    @staticmethod
    def get_last_n_month(year:int ,month:int,n:int) : 
        if month < n    :
            return month+12-n
        else :
            return     month-n
    
    @staticmethod
    def get_last_year(year:int ,month:int,n:int) : 
        if month < n    :
            return year-1
        else :
            return year 