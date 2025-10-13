# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:08:47 2022

@author: dWX1065688
"""

import numpy as np
#import pandas as pd  

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import modules_nn
import utils_nn as utils
import support_lib as sl

import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
from optuna.trial import TrialState

from time import perf_counter

import gc        

''' Clean cache '''
torch.cuda.empty_cache()
gc.collect()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(net, dataset, epoch_num, save_results=False, dir_save='', optimizer=None, scheduler=None, learning_rate=3e-4):
    weight_decay = 0 #1e-2
    # Loss function
    loss_fn = modules_nn.MSELoss(reduction='sum')
    # Algorithm SGD adjustment
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.0)
    if optimizer == None:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, \
        patience=epoch_num, threshold=1e-2, threshold_mode='abs')
    lrs = []
    losses = np.zeros((epoch_num,), dtype=float)
    # Initialize directory for saving results:
    if dir_save == '':
        dir_save = curr_dir
    if save_results == True:
        ''' Save weights before adaptation '''
        net.save_weights(dir_save+'/init_')
    # Train loop
    dp = 0
    t1 = perf_counter()
    for epoch in range(epoch_num):
        loss_aver = 0
        divider_aver = 0
        t_epoch_start = perf_counter()
        for indx in range(dataset.batch_num):
            x_batch, y_batch = dataset.input_batches[indx, :], dataset.target_batches[indx, :]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = x_batch
            y_pred = net.forward(x_batch)
            # print(x_batch.size())
            # print(y_batch.size())
            # print(y_pred.size())
            # sys.exit()
            loss = loss_fn(y_pred, y_batch)
            divider = utils.norm2(y_batch)
            loss_aver += loss.item()
            divider_aver += divider.item()
            loss.backward()
            optimizer.step()
        t_learn_end = perf_counter()
        print('Epoch learn ended: ', t_learn_end - t_epoch_start, 's')
        with torch.no_grad():
            y_pred_full = torch.tensor([]).to(device)
            y_pred_full = net.forward(dataset.input_full.to(device)).detach()
            d_full = dataset.target_full.to(device).detach()
            noise_floor = dataset.noise_floor.to(device).detach()
            d = d_full.cpu().numpy()
            y = y_pred_full.cpu().numpy()
            nf = noise_floor.cpu().numpy()
            NMSEloss_full_dB = sl.nmse_nf(d, d-y, nf)
            losses[epoch] = float(NMSEloss_full_dB)
            if save_results == True:
                np.save(dir_save+'/d.npy', d.reshape(d.shape[1]))
                np.save(dir_save+'/y.npy', y.reshape(y.shape[1]))
                np.save(dir_save+'/nf.npy', nf)
                np.save(dir_save+'/losses.npy', losses)
                ''' Save weights '''
                net.save_weights(dir_save+'/')
            scheduler.step(NMSEloss_full_dB)
            lrs.append(optimizer.param_groups[0]['lr'])
            t_epoch_finish = perf_counter()
            print(f'Epoch = {epoch}, Loss = {NMSEloss_full_dB.item()} dB, Elapsed time: {t_epoch_finish - t_epoch_start}')
    t2 = perf_counter()
    print('Learning time: %.3f' % (t2-t1))
    # net.param_watch()