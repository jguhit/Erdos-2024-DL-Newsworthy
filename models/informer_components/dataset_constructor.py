import torch, torchvision
import pandas as pd
import numpy as np
import sklearn.preprocession as MinMaxScaler
from torch.utils.data import Dataset, DataLoader
''' This coontains the functions needed to prepare the csv file into an input into the transformer model. This includes:
        splitting into train.val, test
        making the data into a tensor
        creating dataset class
        batching the data
'''


#prone to bugs. requires correct formatting for dates
def train_splitter(df, val_start = '2022-09-15', test_start = '2023-06-15'):
        df['stock_time'] = pd.to_datetime(df['stock_time'])
        train = df[df['stock_time'] <= val_start]
        val = df[df['stock_time'] > val_start & df['stock_time']<=test_start]
        test = df[df['stock_time'] > test_start]
        scaler = MinMaxScaler()
        to_scale = ['open', 'high', 'low', 'close', 'volume', 'numtrades', 'vwap']
        train[to_scale] = scaler.fit_transform(train[to_scale])
        val[to_scale] = scaler.transform(val[to_scale])
        test[to_scale] = scaler.transform(test[to_scale])
        return train, val, test

def tensorify(df):
        


class TimeData(Dataset):
        def __init__(self, data, seq_len, pred_len):
                self.data = data
                self.seq_len = seq_len
                self.pred_len = pred_len

        def __len__(self):
                return len(self.data)-self.seq_len-self.pred_len
        
        def __getitem__(self, idx):

