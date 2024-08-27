''' This is based on the exp_infomer model from Zhou et al 2020, adapted to our purposes'''
#have not yet implemented earlystopping
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_constructor as dc
import time
import os 
import sys
sys.path.append('./Informer2020')
from Informer2020.models import model

class Exp_Infomer():
        def __init__(self, args, device):
                self.device = device
                self.args = args
                self.model = _build_model.to(self.device)
                
        def _build_model(self):
             model = Informer(
                             self.args.enc_in,
                             self.args.dec_in,
                             self.args.c_out,
                             self.args.seq_len,
                             self.args.label_len,
                             self.args.pred_len,
                             self.args.factor,
                             self.args.d_model,
                             self.args.n_heads,
                             self.args.e_layers,
                             self.args.d_layers,
                             self.args.d_ff,
                             self.args.dropout,
                             self.args.attn,
                             self.args.embed,
                             self.args.freq,
                             self.args.activation,
                             self.args.output_attention,
                             self.args.distil,
                             self.args.mix,
                             self.device
                                ).float()
                    
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids = self.args.device_ids)
            return model

    def _get_data(self):
            args = self.args
            tick = args.ticker

            df = pd.read_csv('./csv_files/'+tick+'_core.csv')
            train, val, test = dc.train_splitter(df)
            t_train = dc.tensorify(train)
            t_val = dc.tensorify(val)
            t_test = dc.tensorify(test)
            train_loader = DataLoader(t_train, batch_size =args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)
            val_loader = DataLoader(t_val, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)
            test_loader = DataLoader(t_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, drop_last = False)

            return train_loader, val_loader, test_loader

#maybe make this more customizable
    def _select_optimizer(self):
            optimizer = AdamW(model.parameters(), lr = self.args.learning_rate, betas = (.9, .95), weight_decay = self.args.weight_decay)
            return optimizer

    def __select_criterion(self):
            criterion = nn.MSELoss()
            return criterion

    def vali(self, val_loader, criterion):
            self.model.eval()
            total_loss = []
            for i, (seq_X, time_X, seq_y, time_y) in enumerate(val_loader):
                pred, true = self._process_one_batch(
                                    seq_X, time_X, seq_y, time_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
            total_loss = np.average(total_loss)
            self.model.train()
            return total_loss
    
    def train(self, setting):
            train_loader, val_loader, test_loader = self._get_data()
            path = os.path.join('./model_saves',setting)
            if not os.path.exists(path):
                    os.makedirs(path)

            time_now = time.time()
            train_steps = len(train_loader)/args.batch_size

            optimizer = self._select_optimizer()
            criterion = self._select_criterion()
            best_loss = 100
            stop_count = 0
            for epoch in range(self.args.train_epochs):
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, seq_X, time_X, seq_y, time_y in enumerate(train_loader):
                    optimizer.zero_grad()
                    pred_true = self._process_one_batch(seq_X, time_X, seq_y, time_y)
                    loss = criterion(pred, true)
                    train_loss.append(loss.item())

                    if (i+1)%100 == 0:
                        print(f'epoch: {epoch+1},loss: {loss.item()}')
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epcoh)+(train_steps - i))
                        print(f'time remaining: {left_time}')
                        time_now = time.time()

                    loss.backward()
                    optimizer.step()
                print(f'epoch {epoch+1} done')
                train_loss = np.average(train_loss)
                val_loss = self.vali(val_loader, criterion)
                test_loss = self.vali(test_loader, criterion)
                print(f'Epoch {epoch +1}, train_loss: {train_loss}, val_loss: {val_loss}, test_loss: {test_loss}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    stop_count = 0
                else:
                    stop_count +=1
                if stop_count >3:
                    print('Early Stopping')
                    break
            now = str(datetime.datetime.now())
            best_model_path = path +'/'+now+'checkpoint.pth'
            torch.save(model.state_dict(), best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
            return self.model


    def test(self, setting):
            _, _, test_loader = self._get_data()
            self.model.eval()
            
            preds = []
            trues = []
            for i, seq_X, time_X, seq_y, time_y in enumerate(test_loader):
                pred, true = self._process_one_batch(
                                seq_X, time_X, seq_y, time_y)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

            preds = np.array(preds)
            trues = np.array(trues)
            
            accuracy = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
            
            folder_path = /.
