# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:57:16 2022

@author: dWX1065688
"""

import sys
import numpy as np
import scipy.signal as signal
import typing as tp
import torch
import torch.nn as nn
import matplotlib as plt
from time import perf_counter
from torch.utils.data import Dataset
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
import optuna
import utils_nn as utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FIR(nn.Module):
    def __init__(self, num_taps, weights_init=torch.tensor([]), dtype=torch.cfloat):
        super(FIR, self).__init__()
        self.num_taps = num_taps
        self.weights_init = weights_init
        self.taps = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=num_taps, padding='same', dtype=dtype, bias=False)
        self.apply(self._init_weights)
    def forward(self, x):
        x = x.squeeze(dim=1)
        x = torch.flip(x, dims=[2])
        x = self.taps(x)
        x = torch.flip(x, dims=[2])
        x = x.reshape(x.shape[0], x.shape[2])
        return x
    def _init_weights(self, module):
        if len(self.weights_init) == 0:
            self.taps.weight.data.zero_()
            filt_size = self.taps.weight.data.size()[2]
            central_tap = int(filt_size//2)
            self.taps.weight.data[:, :, central_tap] = 1
        else:
            self.taps.weight.data = self.weights_init
            self.taps.requires_grad_(self.weights_init.requires_grad)
        return None
    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
            print(param)
        return None            
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None 
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        assert (reduction == 'sum') or (reduction == 'mean') or (reduction == None), \
            f"Parameter reduction must equal 'sum', 'mean' or None, but '{reduction}' is given"
        self.reduction = reduction
    def forward(self, output, target):
        e = target - output
        if self.reduction == 'sum':
            loss = torch.sum(torch.abs(e)**2)
        if self.reduction == 'mean':
            loss = torch.mean(torch.abs(e)**2, dim=1)
            loss = torch.mean(loss)
        if self.reduction == None:
            loss = torch.sum(torch.abs(e)**2, dim=1)
        return loss

class CTanh(nn.Module):
    def __init__(self):
        super(CTanh, self).__init__()
    def forward(self, x):
        return torch.tanh(x)

class cos(nn.Module):
    def __init__(self):
        super(cos, self).__init__()
    def forward(self, x):
        return torch.cos(x)

class arccos(nn.Module):
    def __init__(self):
        super(arccos, self).__init__()
    def forward(self, x):
        return torch.arccos(x)

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        cond_real = not (x.real >= 0)
        cond_imag = not (x.imag >= 0)
        x[cond_real] = 0+0j
        x[cond_imag] = 0+0j
        return x

class CPReLU(nn.Module):
    def __init__(self):
        super(CPReLU, self).__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1, dtype=torch.cfloat), requires_grad=True)
    def forward(self, x):
        return nn.ReLU(torch.abs(x)+self.bias)*torch.exp(1j*torch.angle(x))

class ScaleShift(nn.Module):
    def __init__(self, channel_num, dim, stype):
        super(ScaleShift, self).__init__()
        self.dim = dim
        self.channel_num = channel_num
        self.scale = torch.nn.Parameter(torch.randn(channel_num, dtype=stype), requires_grad=True)
        self.shift = torch.nn.Parameter(torch.randn(channel_num, dtype=stype), requires_grad=True)
    def forward(self, x):
        if self.dim == '1d':
            shift = self.shift.expand(x.size()[0], x.size()[2], self.channel_num)
            shift = torch.permute(shift, (0, 2, 1))
            scale = self.scale.expand(x.size()[0], x.size()[2], self.channel_num)
            scale = torch.permute(scale, (0, 2, 1))
            x *= scale
            x += shift
        if self.dim == '2d':
            shift = self.shift.expand(x.size()[0], x.size()[2], x.size()[3], self.channel_num)
            shift = torch.permute(shift, (0, 3, 1, 2))
            scale = self.scale.expand(x.size()[0], x.size()[2], x.size()[3], self.channel_num)
            scale = torch.permute(scale, (0, 3, 1, 2))
            x *= scale
            x += shift
        return x

class Identity(nn.Module):
    '''
        Empty class that returns input tensor
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

activat_func_names = {
    'ctanh':  CTanh(),
    'CReLU':  CReLU(),
    'CPReLU': CPReLU(),
    'cos':    cos(),
    'arccos': arccos()
}

feature_func_names = {
    'abs':    np.abs,
    'real':   np.real,
    'imag':   np.imag,
    'same':   utils.same_value
}