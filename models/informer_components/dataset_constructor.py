import torch, torchvision
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
''' This contains the functions needed to prepare the csv file into an input into the transformer model. This includes:
        splitting into train.val, test
        making the data into a tensor
        creating dataset class
        batching the data
'''


#prone to bugs. requires correct formatting for dates
def train_splitter(df, val_start = '2022-09-15', test_start = '2023-06-15'):
        df['stock_time'] = pd.to_datetime(df['stock_time'])
        df.set_index('stock_time', inplace = True)
        df['encoded_sentiment'] = df['encoded_sentiment'].apply(literal_eval)
        train = df[df.index <= val_start]
        val = df[(df.index > val_start) & (df.index<=test_start)]
        test = df[df.index > test_start]
        scaler = MinMaxScaler()
        to_scale = ['open', 'high', 'low', 'close', 'volume', 'numtrades', 'vwap']
        train[to_scale] = scaler.fit_transform(train[to_scale])
        val[to_scale] = scaler.transform(val[to_scale])
        test[to_scale] = scaler.transform(test[to_scale])
        return train, val, test

def tensorify(df, sentiment_sum = True):
        if sentiment_sum:
                df['encoded_sentiment'] = df['encoded_sentiment'].apply(sum)
                df = df.map(lambda x:torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        else:
                cols = list(df.drop(columns = 'encoded_sentiment').columns)
                df['encoded_sentiment'] = df['encoded_sentiment'].apply(lambda x: torch.tensor(x, dtype = torch.float32))
                df[cols] = df.drop(columns = 'encoded_sentiment').map(lambda x:torch.tensor(x, dtype = torch.float32).unsqueeze(0))
        rows = [torch.cat([df[col][i] for col in df.columns]) for i in range(len(df))]
        final = torch.stack(rows)
        return final


#need dataframe to come with columns same order as those in csv files. Otherwise, need to adjust the iloc values in definition
class TimeData(Dataset):
        def __init__(self, data, seq_len,label_len,  pred_len ):
                self.data = data
                self.seq_len = seq_len
                if label_len > seq_len:
                    self.label_len = seq_len
                else:
                    self.label_len = label_len
                self.pred_len = pred_len
                

        def __len__(self):
                return len(self.data)-self.seq_len-self.pred_len
        
        def __getitem__(self, idx):
                if idx + self.seq_len + self.pred_len > len(self.data):
                    raise IndexError('index out of bounds')
                total = self.data[idx:idx+self.seq_len+self.pred_len]
                age_tensor = torch.arange(self.seq_len+self.pred_len).float().unsqueeze(1)/(self.pred_len+self.seq_len-1)
                age_tensor = age_tensor - age_tensor[self.seq_len-1]
                total = torch.cat((total, age_tensor), dim = 1)
                
                X = total[:self.seq_len]
                seq_X = X[:,:-6]
                time_X = X[:, -6:]
                Y = total[self.seq_len - self.label_len:self.seq_len + self.pred_len]
                seq_y = Y[:, 3].unsqueeze(-1)
                time_y = Y[:,-6:]
                return seq_X, time_X, seq_y, time_y
#need to manually adjust seq_y to right 'true' values
