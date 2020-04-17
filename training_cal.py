import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch import optim


from dataloader_actual import CAL_Dataset
from net_actual import get_model
from dataloader_actual import get_data, get_mini_data
from train_actual import fit, custom_loss, validate
from metrics_actual_changed import calc_metrics

# paths
data_path = './Episodes/'
#data_path

if not os.path.exists('models'):
	os.mkdir('models')

if not os.path.exists('total_models'):
	os.mkdir('total_models')


params = {'name': 'model_train_try', 'type_': 'LSTM', 'lr': 1e-4, 'n_h': 100, 'p':0.44, 'seq_len':10}
model, opt = get_model(params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_size = 10
train_dl, valid_dl= get_data(data_path, model.params.seq_len, batch_size)

model, val_hist = fit(1, model, custom_loss, opt, train_dl, valid_dl)
