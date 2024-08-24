import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gluonts.time_feature import time_features_from_frequency_str


def sentiment_to_value(txt:str,neutral_weight = 0):
    if txt == 'positive':
            return 1
    elif txt == 'negative':
            return -1
    elif txt == 'neutral':
            return neutral_weight
    else:
            return 0

def articles(tick:str):
        df = pd.read_csv('../../prebuilt_rob_data/'+tick+'_prerob.csv')
        df = df[df['Publishing Time'].notna()][['Publishing Time', 'Source','rob_sentiment']]#, 'Ticker','Sector']]
        df['Publishing Time'] = pd.to_datetime(df['Publishing Time'])
        return df

def stock_data(tick:str):
        df_stock = pd.read_csv('../../Intraday_StockData/'+tick+'_intraday.csv')
        df_stock['timestamp'] = pd.to_datetime(df_stock['timestamp'])
        df_stock.set_index('timestamp', inplace = True)
        core = pd.read_csv('core_times.csv')
        core['0'] = pd.to_datetime(core['0'])
        return df_stock.filter(items = core['0'], axis = 0).reset_index()

def stock_split_fix(tick:str, df):
        times = {'AAPL':'2020-08-31', 'NVDA':'2021-07-20','AMZN':'2022-06-06','GOOGL':'2022-07-18'}
        if tick not in times:
                return df
        relevant_cols = ['open', 'high', 'low','close'] 
        values = {'AAPL':4, 'NVDA':4, 'AMZN':20, 'GOOGL':20}
        df.loc[df['stock_time']>times[tick], relevant_cols]*=values[tick]
        return df

def core_frame(tick:str,fix_split = True):
        df_stock = stock_data(tick)
        df_articles = articles(tick)
        df_articles = df_articles.sort_values('Publishing Time')
        df_stock = df_stock.sort_values('index')
        df_for = pd.merge_asof(df_articles, df_stock, right_on ='index', left_on='Publishing Time', direction = 'forward')
        merge_col = list(df_stock.columns)
        df_core = pd.merge(df_for, df_stock, on = merge_col, how = 'outer')
        df_core.rename(columns = {'index':'stock_time'}, inplace = True)
        if fix_split:
            df_core = stock_split_fix(tick, df_core)
       # df_core.set_index('stock_time', inplace = True)
        return df_core

#get_dummies returns source to array conversion
def encoded_source(df):
        one_hot_df = pd.get_dummies(df['Source'])
        ohd_array = one_hot_df.apply(lambda x: x.values.astype(int), axis=1)
        df['Source'] = ohd_array
        return df


