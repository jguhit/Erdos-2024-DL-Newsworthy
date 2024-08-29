''' Adapted from main_informer in the Infomer Github by Zhou et al'''
import argparse
import os 
import torch

from exp_informer import Exp_Informer as Exp

parser = argparse.ArgumentParser(description = 'Running Informer')

parser.add_argument('--ticker', type = str, default = 'AAPL', help = 'ticker for stock you want to model')
#frequency is sec here to accomodate 6 features. Need to adapt code for more control
parser.add_argument('--freq', type = str, default = 's', help = 'frequency of time features.')
parser.add_argument('--seq_len', type= int, default = 78, help = 'input sequence length')
parser.add_argument('--label_len', type = int, default = 26, help = 'how many past steps to put into decoder')
parser.add_argument('--pred_len', type = int, default=26, help= 'prediction length')

parser.add_argument('--enc_in', type = int, default = 9, help ='encoder input size, non-time related features')
parser.add_argument('--dec_in', type = int, default = 1, help= 'decoder input size')
parser.add_argument('--c_out', type = int, default = 1, help = 'output size')
parser.add_argument('--d_model', type = int, default = 512, help = 'dimension of model, like tokenizer')
parser.add_argument('--n_heads', type = int, default = 8, help = 'attention heads')
parser.add_argument('--e_layers', type = int, default = 2, help = 'encoder layers')
parser.add_argument('--d_layers', type = int, default = 1, help = 'decoder layers')
parser.add_argument('--s_layers', type = str, default = '3,2,1', help ='num of stack encoders. Dont understand, dont change')
parser.add_argument('--d_ff', type = int, default = 2048, help = 'dimension of fcn?')
parser.add_argument('--factor', type = int, default = 5, help = 'probsparse attn factor, need to understand probsparse')
parser.add_argument('--padding', type = int, default = 0, help = 'what to pad decoder input with. Either 1 or 0')
parser.add_argument('--distil', type = int, default = True, help= 'if you want to use distilling in model')
parser.add_argument('--dropout', type = float, default = .05, help='dropout factor')
parser.add_argument('--attn', type = str, default = 'prob', help = 'attention type used in decoder')
parser.add_argument('--embed', type = str, default = 'timeF', help = 'use time features in encoding? not 100% sure, do not change')
parser.add_argument('--activation', type = str, default ='gelu', help = 'activation function')
parser.add_argument('--output_attention',type = bool, default = False, help='not implemented currently')
parser.add_argument('--mix', type = bool, default = True, help = 'use mix attention in decoder')
parser.add_argument('--num_workers', type = int, default = 0, help = 'dataloader num_workers')
parser.add_argument('--train_epochs', type = int, default = 6, help = 'how many epochs to train')
parser.add_argument('--batch_size', type = int, default = 16, help = 'dataloader batch size')
parser.add_argument('--learning_rate', type = float, default = 1e-4, help='optimizer learning rate')
parser.add_argument('--weight_decay', type = float, default = 1e-1, help = 'optimizer weight decay')
parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
parser.add_argument('--use_multi_gpu', type = bool, default = False, help= 'use multiple gpus')
parser.add_argument('--device_ids', type = str, default = '0,1', help = 'device ids of multiple gpus')

args = parser.parse_args()
args.s_layers = [int(s) for s in args.s_layers.replace(' ','').split(',')]

device = torch.device('mps')
tick = args.ticker
exp = Exp(args, device)
exp.train(setting = 'test_'+tick)
exp.test(setting = 'test_'+tick)
