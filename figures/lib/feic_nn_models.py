# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:57:16 2022

@author: dWX1065688
"""

import sys
import numpy as np
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
import modules_nn
import utils_nn as utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Model_pruning(net, conv2d_ratio, lin_ratio):
    for name, module in net.named_modules():
        # if isinstance(module, torch.nn.Conv2d):
        #     prune.l1_unstructured(module, name='weight', amount=conv2d_ratio)
        #     prune.remove(module, 'weight')
        #     module.weight[module.weight == 0].requires_grad_(False)
        #     # module.weight.requires_grad_(False)
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=lin_ratio)
            # print(list(module.named_buffers()))
            prune.remove(module, 'weight')
            # print(list(module.named_parameters()))
            # module.weight.detach_()
            # module.weight[module.weight == 0].requires_grad_(False)
            module.weight[module.weight == 0].detach_()
    # print(net.state_dict().keys())
    return net

class RVTDCNN(nn.Module):

    def __init__(self, sample_size: int, 
            conv_mem_kernel_size_row: int, conv_mem_kernel_size_col: int,
            conv_mem_in_channel: int, conv_mem_out_channel: int,
            fc_num_layers: int, fc_out_feat: int,
            conv_leak_kernel_size_row: int, conv_leak_out_channel: int,
            p_drop_fc: int, 
            activate: str, is_activate_leak: int,
            batch_norm_mode: str,
            input_feature_num: int):
        super(RVTDCNN, self).__init__()
        self.input_feature_num = input_feature_num
        self.activate = activate
        if self.activate == 'tanh':
            self.activate_fn = nn.Tanh()
        if self.activate == 'relu':
            self.activate_fn = nn.ReLU()
        assert self.activate == 'tanh' or self.activate == 'relu', \
            "Activation function must be tanh or ReLU"
        # Memory convolution initialization
        self.conv_mem_out_channel = conv_mem_out_channel
        self.conv_mem_kernel_size_col = conv_mem_kernel_size_col
        self.conv_mem_out_num_row = (input_feature_num - conv_mem_kernel_size_row + 1)*conv_mem_out_channel*conv_mem_in_channel
        self.conv_mem_out_num_col = (sample_size - conv_mem_kernel_size_col + 1)*conv_mem_out_channel*conv_mem_in_channel
        self.conv_mem_out_num_col_single_ch = sample_size - conv_mem_kernel_size_col + 1
        if batch_norm_mode == 'common':
            batch_norm_mem = nn.BatchNorm2d(self.conv_mem_out_channel)
        if batch_norm_mode == 'simple':
            batch_norm_mem = ScaleShift(self.conv_mem_out_channel, '2d', stype=float)
        self.convolv_mem_layer = nn.Sequential(
            nn.Conv2d(in_channels=conv_mem_in_channel, out_channels=conv_mem_out_channel,
                            kernel_size=(conv_mem_kernel_size_row, conv_mem_kernel_size_col)),
            batch_norm_mem,
            self.activate_fn
        )
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        fc_in_feat = self.conv_mem_out_num_row
        self.fc_out_feat = fc_out_feat
        self.p_drop_fc = p_drop_fc
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(nn.Linear(fc_in_feat, fc_out_feat))
            if batch_norm_mode == 'common':
                self.fc_batchnorm.append(nn.BatchNorm1d(fc_out_feat))
            if batch_norm_mode == 'simple':
                self.fc_batchnorm.append(ScaleShift(fc_out_feat, '1d'), stype=float)
            fc_in_feat = fc_out_feat
        # Leakage path convolution initialization
        self.conv_leak_out_channel = conv_leak_out_channel
        self.conv_leak_out_num_row = (fc_out_feat - conv_leak_kernel_size_row + 1)*conv_leak_out_channel
        if batch_norm_mode == 'common':
            batch_norm_leak = nn.BatchNorm2d(self.conv_leak_out_channel)
        if batch_norm_mode == 'simple':
            batch_norm_leak = ScaleShift(self.conv_leak_out_channel, '2d', stype=float)
        conv_leak = nn.Conv2d(in_channels=1, out_channels=conv_leak_out_channel,
                                kernel_size=(conv_leak_kernel_size_row, self.conv_mem_out_num_col_single_ch))
        conv_leak_layers = [conv_leak, batch_norm_leak]
        if is_activate_leak == 1:
            conv_leak_layers.append(self.activate_fn)
        else: pass
        self.convolv_leak_layer = nn.Sequential(*conv_leak_layers)
        # FC output layer
        self.fc_out = nn.Linear(in_features=self.conv_leak_out_num_row, out_features=2)

    def forward(self, x, mode):
        '''
            mode: 'test' or 'learn'. mode influences on dropout usage
            h: hidden state - stores information from previous batches
        '''
        x_curr = x
        # Memory convolution layer
        x_curr = self.convolv_mem_layer(x_curr)
        # print(x_curr.size())
        # sys.exit()
        x_curr = x_curr.view(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2], x_curr.shape[3])
        # print(x_curr.size())
        x_curr = torch.permute(x_curr, (0, 2, 1))
        # print(x_curr.size())
        # FC-layers
        for fc_layer_i in range(np.size(self.fc_layers)):
            if mode == 'learn':
                x_curr = self.fc_dropout[fc_layer_i](x_curr)
            if mode == 'test':
                x_curr *= (1 - self.p_drop_fc)
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            # print(x_curr.size())
            x_curr = torch.permute(x_curr, (0, 2, 1))
            # print(x_curr.size())
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            # print(x_curr.size())
            x_curr = self.activate_fn(x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
        # Leakage path convolution layer
        x_curr = x_curr.reshape(x_curr.shape[0], 1, x_curr.shape[2], x_curr.shape[1])
        # print(f'after reshape: {x_curr.size()}')
        x_curr = self.convolv_leak_layer(x_curr)
        # print(x_curr.size())
        x_curr = x_curr.view(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2]*x_curr.shape[3])
        # print(x_curr.size())
        # Output layer
        x_curr = self.fc_out(x_curr)
        # sys.exit()
        return x_curr

    def calculate_resources(self):
        # Resources required by first convolutional layer
        K_mem = self.conv_mem_out_channel
        S_mem = self.input_feature_num
        M_mem = self.conv_mem_kernel_size_col
        conv_mem_mults = K_mem*S_mem*M_mem
        conv_mem_adds = conv_mem_mults
        conv_mem_activates = self.input_feature_num*self.conv_mem_out_channel

        # Resources required by second convolutional layer
        K_leak = self.conv_leak_out_channel
        S_leak = self.fc_out_feat
        M_leak = self.conv_mem_out_num_col_single_ch
        conv_leak_mults = K_leak*S_leak*M_leak
        conv_leak_adds = conv_leak_mults
        conv_leak_activates = K_leak*S_leak

        # Resources required by all FC-layers
        Q = conv_mem_activates
        P = self.fc_out_feat
        L = self.fc_num_layers
        fc_mults = P*Q+(P**2)*(L-1) + conv_leak_activates*2
        fc_adds = fc_mults
        fc_activates = P*L+2

        # Resources required by batch-norm parameters
        bn_mults = K_mem*S_mem+K_leak*M_leak+P*L
        bn_adds = bn_mults

        # Resources required by whole model
        total_mults = conv_mem_mults+fc_mults+conv_leak_mults+bn_mults
        total_adds = conv_mem_adds+fc_adds+conv_leak_adds+bn_adds
        total_activates = conv_mem_activates+fc_activates+conv_leak_activates

        print(f'conv_mem_mults: {conv_mem_mults}')
        print(f'conv_mem_adds: {conv_mem_adds}')
        print(f'conv_mem_activates: {conv_mem_activates}')
        print(f'fc_mults: {fc_mults}')
        print(f'fc_adds: {fc_adds}')
        print(f'fc_activates: {fc_activates}')
        print(f'conv_leak_mults: {conv_leak_mults}')
        print(f'conv_leak_adds: {conv_leak_adds}')
        print(f'conv_leak_activates: {conv_leak_activates}')
        print(f'bn_mults: {bn_mults}')
        print(f'bn_adds: {bn_adds}')
        print(f'total_mults: {total_mults}')
        print(f'total_adds: {total_adds}')
        print(f'total_activates: {total_activates}')
        return None

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model peremeter number:', sum.item())
        return sum.item()

    def get_weights(self):
        params_all = []
        names_all = []
        for name, param in self.named_parameters():
            params_all.append(param) 
            names_all.append(name)
        return tuple(names_all), tuple(params_all)
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class Hammerstein_extended(nn.Module):

    def __init__(self, sample_size: int,
            fc_num_layers: int, fc_out_feat: int,
            conv_kernel_size_row: int, conv_out_channel: int,
            p_drop_fc: int, 
            activate: str, is_activate_leak: int,
            batch_norm_mode: str,
            input_feature_num: int):
        super(Hammerstein_extended, self).__init__()
        self.sample_size = sample_size
        self.activate = activate
        if self.activate == 'tanh':
            self.activate_fn = nn.Tanh()
        if self.activate == 'relu':
            self.activate_fn = nn.ReLU()
        assert self.activate == 'tanh' or self.activate == 'relu', \
            "Activation function must be tanh or ReLU"
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        fc_in_feat = input_feature_num
        fc_out_feat = fc_out_feat
        self.p_drop_fc = p_drop_fc
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(nn.Linear(fc_in_feat, fc_out_feat))
            if batch_norm_mode == 'common':
                self.fc_batchnorm.append(nn.BatchNorm1d(fc_out_feat))
            if batch_norm_mode == 'simple':
                self.fc_batchnorm.append(ScaleShift(fc_out_feat, '1d', stype=float))
            fc_in_feat = fc_out_feat
        # Leakage path convolution initialization
        self.conv_out_channel = conv_out_channel
        self.conv_out_num_row = (fc_out_feat - conv_kernel_size_row + 1)*conv_out_channel
        if batch_norm_mode == 'common':
            batch_norm_leak = nn.BatchNorm2d(self.conv_out_channel)
        if batch_norm_mode == 'simple':
            batch_norm_leak = ScaleShift(self.conv_out_channel, '2d', stype=float)
        conv_leak_layers = [nn.Conv2d(in_channels=1, out_channels=conv_out_channel,
                                kernel_size=(conv_kernel_size_row, sample_size)),
                            batch_norm_leak]
        if is_activate_leak == 1:
            conv_leak_layers.append(self.activate_fn)
        else: pass
        self.convolv_leak_layer = nn.Sequential(*conv_leak_layers)
        # FC output layer
        self.fc_out = nn.Linear(in_features=self.conv_out_num_row, out_features=2)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.zero_()
            ind_ones = int(self.sample_size/2) - ((self.sample_size+1) % 2)
            module.weight.data[:, 0, :, ind_ones] = 1
            if module.bias is not None:
                module.bias.data.zero_()
        return None

    def forward(self, x):
        '''
            mode: 'test' or 'learn'. mode influences on dropout usage
            h: hidden state - stores information from previous batches
        '''
        x_curr = x
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[3], x_curr.shape[2])
        # FC-layers
        for fc_layer_i in range(np.size(self.fc_layers)):
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            x_curr = self.activate_fn(x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
        # Leakage path convolution layer
        x_curr = x_curr.reshape(x_curr.shape[0], 1, x_curr.shape[2], x_curr.shape[1])
        x_curr = self.convolv_leak_layer(x_curr)
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2]*x_curr.shape[3])
        # Output layer
        x_curr = self.fc_out(x_curr)
        return x_curr

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model parameter number:', sum.item())
        return sum.item()

    def get_weights(self):
        params_all = []
        names_all = []
        for name, param in self.named_parameters():
            params_all.append(param) 
            names_all.append(name)
        return tuple(names_all), tuple(params_all)
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class Nonlinearity(nn.Module):
    def __init__(self, fc_num_layers: int, fc_out_feat: int,
            p_drop_fc: int, 
            activate: str,
            batch_norm_mode: str,
            features: int,
            delays: int):
        super(Nonlinearity, self).__init__()
        assert len(activate) == len(fc_out_feat) + 1, \
            "Number of activation functions must be 1 more than number of FC-mapping layers"
        self.activate = activate
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        self.features = features
        self.delays = delays
        self.ind_same = utils.feature2ind('same', features, delays)
        self.ind_abs = utils.feature2ind('abs', features, delays)
        self.fc_in_feat = len(utils.feature2ind('abs', features, delays))
        self.fc_out_feat = fc_out_feat
        self.p_drop_fc = p_drop_fc
        self.init_linear = 0
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        self.fc_activate = nn.ModuleList()
        self.fc_activate.append(modules_nn.activat_func_names[activate[0]])
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(nn.Linear(self.fc_in_feat, self.fc_out_feat[fc_layer_i], dtype=torch.float))
            if batch_norm_mode == 'common':
                self.fc_batchnorm.append(nn.BatchNorm1d(self.fc_out_feat[fc_layer_i]))
            if batch_norm_mode == 'simple':
                self.fc_batchnorm.append(modules_nn.ScaleShift(self.fc_out_feat[fc_layer_i], '1d', stype=torch.float))
            if batch_norm_mode == 'nothing':
                self.fc_batchnorm.append(modules_nn.Identity())        
            self.fc_activate.append(modules_nn.activat_func_names[activate[fc_layer_i+1]])
            self.fc_in_feat = self.fc_out_feat[fc_layer_i]
        # FC output layer
        self.fc_out = nn.Linear(in_features=self.fc_out_feat[fc_layer_i], out_features=1, dtype=torch.cfloat)
        self.apply(self._init_weights)
        # self.param_watch()
        # sys.exit()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module.weight.data.dtype == torch.float32:
            torch.nn.init.uniform_(module.weight.data, 0, self.fc_out_feat[self.init_linear])
            module.weight.data = torch.round(module.weight.data)
            module.weight.data -= int(self.fc_out_feat[self.init_linear]/2)
            if module.bias is not None:
                module.bias.data.zero_()
            self.init_linear += 1
        if isinstance(module, nn.Linear) and module.weight.data.dtype == torch.cfloat:
            torch.nn.init.uniform_(module.weight.data, -1, 1)
            module.weight.data *= 1e-2
            if module.bias is not None:
                module.bias.data.zero_()
        return None
    def init_internal(self, dims):
        self.fc_layers[0].weight.data = utils.init_matrix(dims)
        self.fc_layers[0].weight.data.to(device)
        self.fc_layers[0].weight.data.requires_grad_(True)

    def forward(self, x):
        # print(x.size())
        x_in = x[:, :, self.ind_same, :]
        # print(f'ind_same: {self.ind_same}')
        # print(f'x_in.size: {x_in.size()}')
        x_curr = torch.abs(x[:, :, self.ind_abs, :])
        # print(f'ind_abs: {self.ind_abs}')
        # print(f'x_curr.size: {x_curr.size()}')
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2], x_curr.shape[3])
        x_curr = torch.permute(x_curr, (0, 2, 1))
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2], x_in.shape[3])
        x_in = torch.permute(x_in, (0, 2, 1))
        # print(f'x_in.size after reshape: {x_in.size()}')
        # print(f'x_curr.size after reshape: {x_curr.size()}')
        # FC-layers
        x_curr = self.fc_activate[0](x_curr)
        for fc_layer_i in range(np.size(self.fc_layers)):
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            # print(f'x_curr.size after Linear: {x_curr.size()}')
            x_curr = torch.permute(x_curr, (0, 2, 1))
            # print(f'x_curr.size after permute: {x_curr.size()}')
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            # print(f'x_curr.size after BatchNorm: {x_curr.size()}')
            x_curr = self.fc_activate[fc_layer_i + 1](x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
            # print(f'x_curr.size after permute: {x_curr.size()}')
        # Output layer
        x_curr = x_curr.type(torch.cfloat)
        # print(f'x_curr.size after .to(complex): {x_curr.size()}')
        x_curr = self.fc_out(x_curr)
        # print(f'x_curr.size after Linear_out: {x_curr.size()}')
        x_curr = x_curr * x_in
        # print(f'x_curr.size after mult by signal: {x_curr.size()}')
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2])
        # print(f'x_curr.size after final reshape: {x_curr.size()}')
        return x_curr

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
            print(param)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model parameter number:', sum.item())
        return sum.item()

    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class TDNN(nn.Module):
    '''
        Current model works only on 1 sample rate
    '''
    def __init__(self, fir_tap_num: int,
            fc_num_layers: int, fc_out_feat: int,
            p_drop_fc: int, 
            activate: str,
            batch_norm_mode: str,
            resample_ratio: float,
            fir_ds: np.array,
            features: int,
            delays: int):
        super(TDNN, self).__init__()
        self.nonlin = Nonlinearity(fc_num_layers = fc_num_layers,
                            fc_out_feat = fc_out_feat,
                            p_drop_fc = p_drop_fc,
                            activate = activate,
                            batch_norm_mode = batch_norm_mode,
                            features=features,
                            delays=delays).to(device)
        self.fir_ds = torch.tensor(fir_ds, dtype=torch.cfloat, requires_grad=False).to(device).unsqueeze(dim=0).unsqueeze(dim=0)
        if resample_ratio < 1:
            resample_ratio = 1/resample_ratio
        self.resample_ratio = int(resample_ratio)
        self.num_taps = fir_tap_num
        self.fir = modules_nn.FIR(num_taps=self.num_taps)
    def _init_weights(self, module):     
        return None
    def forward(self, x):
        x_curr = x
        x_curr = self.nonlin(x_curr)
        x_curr = x_curr[::self.resample_ratio, ::self.resample_ratio]
        x_curr = x_curr.unsqueeze(dim=1).unsqueeze(dim=1)
        x_curr = self.fir(x_curr)
        return x_curr

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
            print(param)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model parameter number:', sum.item())
        return sum.item()

    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class TDNN_HSR_FIR_LSR(nn.Module):
    '''
        In current model TDNN works on HSR (High sample rate),
        but FIR works on LSR (Low sample rate)
    '''
    def __init__(self, fir_tap_num: int,
            fc_num_layers: int, fc_out_feat: int,
            p_drop_fc: int, 
            activate: str,
            batch_norm_mode: str,
            resample_ratio: float,
            fir_ds: np.array,
            features: int,
            delays: int):
        super(TDNN_HSR_FIR_LSR, self).__init__()
        self.nonlin = Nonlinearity(fc_num_layers=fc_num_layers,
                            fc_out_feat=fc_out_feat,
                            p_drop_fc=p_drop_fc,
                            activate=activate,
                            batch_norm_mode=batch_norm_mode,
                            features=features,
                            delays=delays).to(device)
        fir_ds = torch.tensor(fir_ds, dtype=torch.cfloat, requires_grad=False).to(device).unsqueeze(dim=0).unsqueeze(dim=0)
        self.fir_ds = modules_nn.FIR(num_taps=fir_ds.shape[2], weights_init=fir_ds)
        if resample_ratio < 1:
            resample_ratio = 1/resample_ratio
        self.resample_ratio = int(resample_ratio)
        self.num_taps = fir_tap_num
        self.fir = modules_nn.FIR(num_taps=self.num_taps)
    def _init_weights(self, module):     
        return None
    def forward(self, x):
        x_curr = x
        # print(x_curr.size())
        x_curr = self.nonlin(x_curr)
        # print(x_curr.size())
        x_curr = x_curr.unsqueeze(dim=1).unsqueeze(dim=1)
        # print(x_curr.size())
        # print(self.fir_ds.taps.weight.requires_grad)
        x_curr = self.fir_ds(x_curr)
        # print(x_curr.size())
        x_curr = x_curr[::self.resample_ratio, ::self.resample_ratio]
        # print(x_curr.size())
        x_curr = x_curr.unsqueeze(dim=1).unsqueeze(dim=1)
        # print(x_curr.size())
        x_curr = self.fir(x_curr)
        # print(x_curr.size())
        # print('End of forward')
        return x_curr
    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
            print(param)
        return None   
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model parameter number:', sum.item())
        return sum.item()
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None       
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None  
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class SIC_AI(nn.Module):

    def __init__(self, sample_size: int,
            fc_num_layers: int, fc_out_feat: int,
            conv_kernel_size_row: int, conv_out_channel: int,
            p_drop_fc: int, 
            activate: str, is_activate_leak: int,
            batch_norm_mode: str,
            input_feature_num: int):
        super(Hammerstein_extended, self).__init__()
        self.sample_size = sample_size
        self.activate = activate
        if self.activate == 'tanh':
            self.activate_fn = nn.Tanh()
        if self.activate == 'relu':
            self.activate_fn = nn.ReLU()
        assert self.activate == 'tanh' or self.activate == 'relu', \
            "Activation function must be tanh or ReLU"
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        fc_in_feat = input_feature_num
        fc_out_feat = fc_out_feat
        self.p_drop_fc = p_drop_fc
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(nn.Linear(fc_in_feat, fc_out_feat))
            if batch_norm_mode == 'common':
                self.fc_batchnorm.append(nn.BatchNorm1d(fc_out_feat))
            if batch_norm_mode == 'simple':
                self.fc_batchnorm.append(ScaleShift(fc_out_feat, '1d', stype=float))
            fc_in_feat = fc_out_feat
        # Leakage path convolution initialization
        self.conv_out_channel = conv_out_channel
        self.conv_out_num_row = (fc_out_feat - conv_kernel_size_row + 1)*conv_out_channel
        if batch_norm_mode == 'common':
            batch_norm_leak = nn.BatchNorm2d(self.conv_out_channel)
        if batch_norm_mode == 'simple':
            batch_norm_leak = ScaleShift(self.conv_out_channel, '2d', stype=float)
        conv_leak_layers = [nn.Conv2d(in_channels=1, out_channels=conv_out_channel,
                                kernel_size=(conv_kernel_size_row, sample_size)),
                            batch_norm_leak]
        if is_activate_leak == 1:
            conv_leak_layers.append(self.activate_fn)
        else: pass
        self.convolv_leak_layer = nn.Sequential(*conv_leak_layers)
        # FC output layer
        self.fc_out = nn.Linear(in_features=self.conv_out_num_row, out_features=2)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.zero_()
            ind_ones = int(self.sample_size/2) - ((self.sample_size+1) % 2)
            module.weight.data[:, 0, :, ind_ones] = 1
            if module.bias is not None:
                module.bias.data.zero_()
        return None

    def forward(self, x):
        '''
            mode: 'test' or 'learn'. mode influences on dropout usage
            h: hidden state - stores information from previous batches
        '''
        x_curr = x
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[3], x_curr.shape[2])
        # FC-layers
        for fc_layer_i in range(np.size(self.fc_layers)):
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            x_curr = self.activate_fn(x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
        # Leakage path convolution layer
        x_curr = x_curr.reshape(x_curr.shape[0], 1, x_curr.shape[2], x_curr.shape[1])
        x_curr = self.convolv_leak_layer(x_curr)
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2]*x_curr.shape[3])
        # Output layer
        x_curr = self.fc_out(x_curr)
        return x_curr

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model parameter number:', sum.item())
        return sum.item()

    def get_weights(self):
        params_all = []
        names_all = []
        for name, param in self.named_parameters():
            params_all.append(param) 
            names_all.append(name)
        return tuple(names_all), tuple(params_all)
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class Hammerstein_extended_cmpx(nn.Module):

    def __init__(self, sample_size: int,
            fc_num_layers: int, fc_out_feat: int,
            conv_kernel_size_row: int, conv_out_channel: int,
            p_drop_fc: int, 
            activate: str, is_activate_leak: int,
            batch_norm_mode: str,
            input_feature_num: int):
        super(Hammerstein_extended_cmpx, self).__init__()
        self.activate = activate
        ACTIVAT_CORRECT = (self.activate == 'ctanh') or (self.activate == 'crelu') or (self.activate == 'cprelu')
        if self.activate == 'ctanh':
            self.activate_fn = CTanh()
        elif self.activate == 'crelu':
            self.activate_fn = CReLU()
        elif self.activate == 'cprelu':
            self.activate_fn = CPReLU()
        else:
            assert ACTIVAT_CORRECT, 'Wrong activation function have been chosen'
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        fc_in_feat = input_feature_num
        fc_out_feat = fc_out_feat
        print(fc_in_feat)
        print(fc_out_feat)
        self.p_drop_fc = p_drop_fc
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(ComplexLinear(fc_in_feat, fc_out_feat))
            if batch_norm_mode == 'common':
                self.fc_batchnorm.append(ComplexBatchNorm1d(fc_out_feat))
            if batch_norm_mode == 'simple':
                self.fc_batchnorm.append(ScaleShift(fc_out_feat, '1d', stype=torch.cfloat))
            fc_in_feat = fc_out_feat
        # Leakage path convolution initialization
        self.conv_out_channel = conv_out_channel
        self.conv_out_num_row = (fc_out_feat - conv_kernel_size_row + 1)*conv_out_channel
        if batch_norm_mode == 'common':
            batch_norm_leak = ComplexBatchNorm2d(self.conv_out_channel)
        if batch_norm_mode == 'simple':
            batch_norm_leak = ScaleShift(self.conv_out_channel, '2d', stype=torch.cfloat)
        conv_leak_layers = [ComplexConv2d(in_channels=1, out_channels=conv_out_channel,
                                kernel_size=(conv_kernel_size_row, sample_size)),
                            batch_norm_leak]
        if is_activate_leak == 1:
            conv_leak_layers.append(self.activate_fn)
        else: pass
        self.convolv_leak_layer = nn.Sequential(*conv_leak_layers)
        # FC output layer
        self.fc_out = ComplexLinear(in_features=self.conv_out_num_row, out_features=1)

    def forward(self, x, mode):
        '''
            mode: 'test' or 'learn'. mode influences on dropout usage
            h: hidden state - stores information from previous batches
        '''
        x_curr = x
        print(x_curr.size())
        x_curr = x_curr.reshape(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[3], x_curr.shape[2])
        print(x_curr.size())
        # FC-layers
        for fc_layer_i in range(np.size(self.fc_layers)):
            if mode == 'learn':
                x_curr = self.fc_dropout[fc_layer_i](x_curr)
            if mode == 'test':
                x_curr *= (1 - self.p_drop_fc)
            # print(x_curr.size())
            # x_curr = x_curr[:, 0, :]
            print(x_curr.size())
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            print(x_curr.size())
            x_curr = torch.permute(x_curr, (0, 2, 1))
            print(x_curr.size())
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            print(x_curr.size())
            x_curr = self.activate_fn(x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
            print(x_curr.size())
        # Leakage path convolution layer
        x_curr = x_curr.reshape(x_curr.shape[0], 1, x_curr.shape[2], x_curr.shape[1])
        print(x_curr.size())
        x_curr = self.convolv_leak_layer(x_curr)
        print(x_curr.size())
        x_curr = x_curr.view(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2]*x_curr.shape[3])
        print(x_curr.size())
        # Output layer
        x_curr = self.fc_out(x_curr)
        # x_curr = x_curr.view(x_curr.shape[0]*x_curr.shape[1])
        x_curr_real = x_curr.real
        x_curr_imag = x_curr.imag
        # print(x_curr_real.size())
        # print(x_curr_imag.size())
        x_curr = torch.cat((x_curr_real, x_curr_imag), dim=1)
        # print(x_curr.size())
        # sys.exit()
        return x_curr

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model peremeter number:', sum.item())
        return sum.item()

    def get_weights(self):
        params_all = []
        names_all = []
        for name, param in self.named_parameters():
            params_all.append(param) 
            names_all.append(name)
        return tuple(names_all), tuple(params_all)
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class HCRNN(nn.Module):  

    def __init__(self, sample_size: int,
                 conv_kernel_size_row: int, conv_kernel_size_col: int, 
                 fc_num_layers: int, fc_out_feat: int,
                 p_drop_fc: int, p_drop_rnn: int, 
                 conv_in_channel: int, conv_out_channel: int,
                 rnn_hidden_dim: int, 
                 activate: str,
                 input_feature_num: int):
        super(HCRNN, self).__init__()
        self.activate = activate
        if self.activate == 'tanh':
            self.activate_fn = nn.Tanh()
        if self.activate == 'relu':
            self.activate_fn = nn.ReLU()
        assert self.activate == 'tanh' or self.activate == 'relu', \
            "Activation function must be tanh or ReLU"
        # Convolution initialization
        self.conv_out_channel = conv_out_channel
        self.conv_out_num_row = (input_feature_num - conv_kernel_size_row + 1)*conv_out_channel*conv_in_channel
        self.conv_out_num_col = (sample_size - conv_kernel_size_col + 1)*conv_out_channel*conv_in_channel
        self.conv_out_num = (input_feature_num - conv_kernel_size_row + 1)*(sample_size - conv_kernel_size_col + 1)*conv_out_channel*conv_in_channel
        self.convolv_layer = nn.Sequential(
            nn.Conv2d(in_channels=conv_in_channel, out_channels=conv_out_channel,
                            kernel_size=(conv_kernel_size_row, conv_kernel_size_col)),
            nn.BatchNorm2d(self.conv_out_channel),
            self.activate_fn
        )
        # FC-layers initialization
        self.fc_num_layers = fc_num_layers
        fc_in_feat = self.conv_out_num_row
        fc_out_feat = fc_out_feat
        self.p_drop_fc = p_drop_fc
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self.fc_batchnorm = nn.ModuleList()
        for fc_layer_i in range(self.fc_num_layers):
            self.fc_dropout.append(nn.Dropout(self.p_drop_fc))
            self.fc_layers.append(nn.Linear(fc_in_feat, fc_out_feat))
            self.fc_batchnorm.append(nn.BatchNorm1d(fc_out_feat))
            fc_in_feat = fc_out_feat
        # RNN initialization
        self.rnn_hidden_dim = rnn_hidden_dim
        self.p_drop_rnn = p_drop_rnn
        self.rnn = nn.RNN(fc_out_feat, rnn_hidden_dim, 1, batch_first=True, nonlinearity=self.activate, dropout=self.p_drop_rnn)
        self.fc_out = nn.Linear(in_features=rnn_hidden_dim, out_features=2)
        # Batchnorm initialization
        self.bn_rnn = nn.BatchNorm1d(rnn_hidden_dim)

    def forward(self, x, h, mode):
        '''
            mode: 'test' or 'learn'. mode influences on dropout usage
            h: hidden state - stores information from previous batches
        '''
        x_curr = x
        # Conv1 layer
        x_curr = self.convolv_layer(x_curr)
        x_curr = x_curr.view(x_curr.shape[0], x_curr.shape[1]*x_curr.shape[2], x_curr.shape[3])
        x_curr = torch.permute(x_curr, (0, 2, 1))
        # FC-layers
        for fc_layer_i in range(np.size(self.fc_layers)):
            if mode == 'learn':
                x_curr = self.fc_dropout[fc_layer_i](x_curr)
            if mode == 'test':
                x_curr *= (1 - self.p_drop_fc)
            x_curr = self.fc_layers[fc_layer_i](x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
            x_curr = self.fc_batchnorm[fc_layer_i](x_curr)
            x_curr = self.activate_fn(x_curr)
            x_curr = torch.permute(x_curr, (0, 2, 1))
        # Recurrent layer     
        x_curr, _ = self.rnn(x_curr)
        x_curr = x_curr[:, x_curr.shape[1]-1, :]
        x_curr = self.bn_rnn(x_curr)
        h = x_curr
        # Output layer
        x_curr = self.fc_out(x_curr)
        return x_curr, h

    def param_watch(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
        return None
    
    def param_number(self):
        sum = 0
        for _, param in self.named_parameters():
            sum += torch.prod(torch.tensor(param.shape))
        print('Whole model peremeter number:', sum.item())
        return sum.item()

    def get_weights(self):
        params_all = []
        names_all = []
        for name, param in self.named_parameters():
            params_all.append(param) 
            names_all.append(name)
        return tuple(names_all), tuple(params_all)
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        return None
        
    def save_weights(self, path=''):
        torch.save(self.state_dict(), path+'weights')
        return None
    
    def load_weights(self, path_name, device='cpu'):
        return torch.load(path_name, map_location=torch.device(device))

class Dataset_FEIC(Dataset):
    def __init__(self, inputs, targets, sample_size: int, target_mode: str, batch_len: int, noise_floor=[], device='cuda',
                 sample_overlap: int=0, input_feature_funcs=None, delays=None, data_type=float, resample_ratio: float=1):
        super(Dataset, self).__init__()
        
        assert sample_size > 0, "Sample size must be positive"
        if np.size(inputs.shape) == 1: 
            inputs = inputs[np.newaxis, np.newaxis, :]
        if np.size(targets.shape) == 1: 
            targets = targets[np.newaxis, np.newaxis, :]
        if np.size(inputs.shape) == 2: 
            inputs = inputs[np.newaxis, :]
        if np.size(targets.shape) == 2: 
            targets = targets[np.newaxis, :]
        assert np.size(inputs.shape) == 3, "Model inputs must be 3-dimensional"
        assert np.size(targets.shape) == 3, "Model targets must be 3-dimensional"
        assert np.size(inputs, axis=1) == np.size(targets, axis=1), \
            "Input array epoch number must equal target array epoch number"
        if target_mode == 'resample':
            assert int(np.size(inputs, axis=2)*resample_ratio) == np.size(targets, axis=2), \
                "Input array size, multiplied by resample ratio must equal target array size"
        assert 0 <= sample_overlap < sample_size, \
            "Sample_overlap must be non-negative and lower than size of batch"
        assert np.size(input_feature_funcs) > 0, \
            "Input feature map must include at least 1 row"
        assert len(input_feature_funcs) == len(delays), \
            "Number of lists in 'delays' must equal number of input features"
        
        self.index = 0
        self.epoch = 0
        self.target_mode = target_mode
        self.sample_size_input = sample_size
        if self.target_mode == 'ones':
            self.sample_size_target = 1
        if self.target_mode == 'vect':
            self.sample_size_target = int(self.sample_size_input * resample_ratio)
        self.sample_overlap = sample_overlap
        self.size_column_input = np.size(inputs, axis=2)
        self.size_column_target = np.size(targets, axis=2)
        self.size_epoch = np.size(inputs, axis=1)
        self.size_channel = np.size(inputs, axis=0)

        # Define batch number for the input and target signal signal
        self.resample_ratio = resample_ratio
        self.batch_len_input = batch_len
        self.batch_len_target = int(self.batch_len_input * self.resample_ratio)
        if self.size_column_input % self.batch_len_input == 0:
            self.batch_num = int(self.size_column_input/self.batch_len_input)
        else:
            self.batch_num = int(np.floor(self.size_column_input/self.batch_len_input))   

        assert (data_type == float) or (data_type == torch.cfloat), \
            f"Data type must float or complex, but {data_type} is given"
        self.data_type = data_type
        if self.data_type == float:
            self.targets = torch.FloatTensor(1, 2, self.size_column_target, self.size_epoch).zero_()
            for epoch in range(self.size_epoch):
                self.targets[0, 0, :, epoch] = torch.FloatTensor(targets[0, epoch, :].real.tolist())
                self.targets[0, 1, :, epoch] = torch.FloatTensor(targets[0, epoch, :].imag.tolist())
            self.inputs = torch.FloatTensor(self.size_channel, utils.len_nested_list(delays), self.size_column_input, self.size_epoch).zero_()
        if self.data_type == torch.cfloat:
            self.targets = torch.tensor(np.zeros((1, 1, self.size_column_target, self.size_epoch)), dtype=torch.cfloat).zero_()
            for epoch in range(self.size_epoch):
                self.targets[0, 0, :, epoch] = torch.tensor(targets[0, epoch, :], dtype=torch.cfloat)
            tmp = torch.FloatTensor(self.size_channel, utils.len_nested_list(delays), self.size_column_input, self.size_epoch).zero_()
            self.inputs = torch.complex(tmp, tmp)

        for channel in range(self.size_channel):
            for epoch in range(self.size_epoch):
                feature_index = 0
                for feat_num, feature in enumerate(input_feature_funcs):
                    feature = modules_nn.feature_func_names[feature]
                    for i_delay, delay in enumerate(delays[feat_num]):
                        inputs_delayed = np.roll(inputs[channel, epoch, :], -1*delay)
                        self.inputs[channel, feature_index, :, epoch] = torch.tensor(feature(inputs_delayed), dtype=data_type)
                        feature_index += 1
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        if noise_floor != []:
            self.noise_floor = torch.tensor(noise_floor).to(device)
        else:
            self.noise_floor = torch.zeros(np.size(inputs), dtype=self.data_type)

        ''' Prepare batches of dataset '''
        input_batches = self.inputs[:, :, :, 0].unfold(2, self.sample_size_input, 1)
        target_batches = self.targets[:, :, :, 0].unfold(2, self.sample_size_target, 1)
        input_batches = torch.permute(input_batches, (2, 0, 1, 3))
        target_batches = target_batches.reshape(target_batches.shape[2], target_batches.shape[3])

        tmp_input = torch.zeros(self.sample_size_input-1, input_batches.size()[1], input_batches.size()[2], input_batches.size()[3])
        tmp_target = torch.zeros(self.sample_size_target-1, target_batches.size()[1])
        input_batches = torch.cat((input_batches, tmp_input), dim=0)
        target_batches = torch.cat((target_batches, tmp_target), dim=0)

        input_batches = input_batches.unfold(0, self.batch_len_input, self.batch_len_input)
        target_batches = target_batches.unfold(0, self.batch_len_target, self.batch_len_target)
        self.input_batches = torch.permute(input_batches, (0, 4, 1, 2, 3)).to(device)
        self.target_batches = torch.permute(target_batches, (0, 2, 1)).to(device)
        assert self.input_batches.shape[0] == self.target_batches.shape[0], \
            f"Batch number of input and target data must equal, but {self.input_batches.shape[0]} and {self.target_batches.shape[0]} are given correspondingly"

        ''' Prepare full dataset '''
        input = self.inputs[:, :, :, 0].unfold(2, 1, 1)
        target = self.targets[:, :, :, 0].unfold(2, 1, 1)

        input = torch.permute(input, (0, 3, 1, 2))
        target = target.reshape(target.shape[3], target.shape[2])

        delta_data = self.input_batches.size()[0]*self.input_batches.size()[1]-input.size()[3]
        if delta_data > 0:
            tmp_input = torch.zeros(input.size()[0], input.size()[1], input.size()[2], self.input_batches.size()[0]*self.input_batches.size()[1]-input.size()[3])
            tmp_target = torch.zeros(target.size()[0], self.target_batches.size()[0]*self.target_batches.size()[1]-target.size()[1])
            self.input_full = torch.cat((input, tmp_input), dim=3).to(device)
            self.target_full = torch.cat((target, tmp_target), dim=1).to(device)
        else:
            self.input_full = input[:self.input_batches.size()[0]*self.input_batches.size()[1], :]
            self.target_full = target[:self.target_batches.size()[0]*self.target_batches.size()[1], :]
        # self.target_full = self.target_full.view(self.target_full.shape[0]*self.target_full.shape[1], self.target_full.shape[2])

    def __whole_len__(self) -> int:       
        return self.size_epoch*((self.size_column - self.sample_overlap) // (self.sample_size - self.sample_overlap))
    
    def __getitems__(self) -> tp.Tuple[torch.Tensor]: 
        return self.input_batches, self.target_batches 

    def __getfull__(self) -> tp.Tuple[torch.Tensor]: 
        return self.input_full, self.target_full 

class Dataset_FEIC_RNN(Dataset):
    def __init__(self, inputs, targets, sample_size: int, \
                 batch_len: int, device='cpu', input_feature_order = [1, 2, 3]):
        super(Dataset, self).__init__()
        
        assert sample_size > 0, "Batch size must be positive"
        if np.size(inputs.shape) == 1: 
            inputs = inputs[np.newaxis, np.newaxis, :]
        if np.size(targets.shape) == 1: 
            targets = targets[np.newaxis, np.newaxis, :]
        if np.size(inputs.shape) == 2: 
            inputs = inputs[np.newaxis, :]
        if np.size(targets.shape) == 2: 
            targets = targets[np.newaxis, :]
        assert np.size(inputs.shape) == 3, "Model inputs must be 3-dimensional"
        assert np.size(targets.shape) == 3, "Model targets must be 3-dimensional"
        assert np.size(inputs, axis=1) == np.size(targets, axis=1), \
            "Input array epoch number must equal target array epoch number"
        assert np.size(inputs, axis=2) == np.size(targets, axis=2), \
            "Input array size must equal target array size"
        
        self.index = 0
        self.epoch = 0
        self.device = device
        self.sample_size = sample_size
        self.size_column = np.size(inputs, axis=2)
        self.size_epoch = np.size(inputs, axis=1)
        self.size_channel = np.size(inputs, axis=0)
        self.batch_len = batch_len
        if self.size_column % batch_len == 0:
            self.batch_num = int(self.size_column/batch_len) - 1
        else:
            self.batch_num = int(np.floor(self.size_column/batch_len))
        assert self.sample_size <= self.batch_num, "Sample size mustn`t excced the length of batch"
        
        self.targets = torch.FloatTensor(1, 2, self.size_column, self.size_epoch).zero_()
        for epoch in range(self.size_epoch):
            self.targets[0, 0, :, epoch] = torch.FloatTensor(targets[0, epoch, :].real.tolist())
            self.targets[0, 1, :, epoch] = torch.FloatTensor(targets[0, epoch, :].imag.tolist())
        self.inputs = torch.FloatTensor(self.size_channel, 1*(np.size(input_feature_order)+2), self.size_column, self.size_epoch).zero_()
        for channel in range(self.size_channel):
            for epoch in range(self.size_epoch):
                self.inputs[channel, 0, :, epoch] = torch.FloatTensor(inputs[channel, epoch, :].real)
                self.inputs[channel, 1, :, epoch] = torch.FloatTensor(inputs[channel, epoch, :].imag)
                for indx, power in enumerate(input_feature_order):
                    self.inputs[channel, 2+indx, :, epoch] = torch.FloatTensor(np.abs(inputs[channel, epoch, :])**power)
        # self.inputs = self.inputs.to(device)
        # self.targets = self.targets.to(device)
    
    def __getitems__(self, epoch=0, rand: bool=False) -> tp.Tuple[torch.Tensor]: 
        assert -self.size_epoch+1 <= epoch <= self.size_epoch-1, \
            "Epoch index is out of the input array"
         
        self.epoch = epoch
        
        input_batches = self.inputs[:, :, :, epoch].unfold(2, self.sample_size, 1)
        target_batches = self.targets[0, :, :-self.sample_size+1, epoch]
        input_batches = torch.permute(input_batches, (2, 0, 1, 3))
        target_batches = torch.permute(target_batches, (1, 0))

        input_batches = input_batches.unfold(0, self.batch_num, self.batch_num)
        target_batches = target_batches.unfold(0, self.batch_num, self.batch_num)
        print(input_batches.size())
        input_batches = torch.permute(input_batches, (4, 0, 1, 2, 3))
        target_batches = torch.permute(target_batches, (2, 0, 1))
        return input_batches, target_batches 

    def __getfull__(self, sig_len: int, epoch=0, rand: bool=False) -> tp.Tuple[torch.Tensor]: 
        assert -self.size_epoch+1 <= epoch <= self.size_epoch-1, \
            "Epoch index is out of the input array"         
        self.epoch = epoch
        input = self.inputs[:, :, :, epoch].unfold(2, self.sample_size, 1)
        target = self.targets[0, :, :-self.sample_size+1, epoch]
        input = torch.permute(input, (2, 0, 1, 3))
        target = torch.permute(target, (1, 0))
        return input, target 

class Dataset_FEIC_cmpx(Dataset):
    def __init__(self, inputs, targets, sample_size: int, batch_len: int, device='cuda',
                 sample_overlap: int=0, input_feature_funcs=None, delays=None):
        super(Dataset, self).__init__()
        
        assert np.size(inputs.shape) == 1, "Model inputs must be 1-dimensional"
        assert np.size(targets.shape) == 1, "Model targets must be 1-dimensional"
        assert np.size(inputs) == np.size(targets), \
            "Input array size must equal target array size"
        assert 0 <= sample_overlap < sample_size, \
            "Sample_overlap must be non-negative and lower than size of batch"
        assert np.size(input_feature_funcs) > 0, \
            "Input feature map must include at least 1 row"
        
        self.index = 0
        self.sample_size = sample_size
        self.sample_overlap = sample_overlap
        self.size_column = np.size(inputs)

        self.batch_len = batch_len
        if self.size_column % batch_len == 0:
            self.batch_num = int(self.size_column/batch_len)
        else:
            self.batch_num = int(np.floor(self.size_column/batch_len))

        self.targets = torch.complex(torch.FloatTensor(targets.real), torch.FloatTensor(targets.imag))
        self.inputs = torch.FloatTensor(1, np.size(input_feature_funcs)*np.size(delays), self.size_column).zero_()
        self.inputs = torch.complex(self.inputs, self.inputs)
        for feat_num, feature in enumerate(input_feature_funcs):
            for i_delay, delay in enumerate(delays):
                inputs_delayed = np.roll(inputs, delay)
                self.inputs[:, feat_num*np.size(delays)+i_delay, :] = feature(torch.complex(torch.FloatTensor(inputs_delayed.real), torch.FloatTensor(inputs_delayed.imag)))
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        
    def __len__(self) -> int:       
        return (self.size_column - self.sample_overlap)// (self.sample_size - self.sample_overlap)
    
    def __whole_len__(self) -> int:       
        return self.size_epoch*((self.size_column - self.sample_overlap) // (self.sample_size - self.sample_overlap))
    
    def _getCnnItem(self, index=0, rand: bool=False) -> tp.Tuple[torch.Tensor]: 
        
        effect_sample_size = self.sample_size-self.sample_overlap
        
        if rand == False:    
            if type(index) == list:
                assert 0 < np.size(index) <= 2, \
                    "List of item batch indices must contain 1 or 2 indices"
                ind_start = index[0]
                if np.size(index) == 1:
                    ind_end = index[0]+1
                else:
                    ind_end = index[1]
            else:
                ind_start = index
                ind_end = index+1
        else:
            index = np.random.randint(int(np.floor((self.size_column-1)/self.sample_size)))
            ind_start = index
            ind_end = index+1

        assert -self.size_column+1 <= ind_start*effect_sample_size <= self.size_column-1, \
            "Batch start index is out of the input array"
        assert -self.size_column+1 <= (ind_end-1)*effect_sample_size <= self.size_column-1, \
            "Batch end index is out of the input array"
         
        self.index = ind_end     
        
        input_batches = torch.tensor([]).to(device)
        target_batches = torch.tensor([]).to(device)
        for index in range(ind_start, ind_end):
            if (index+1)*effect_sample_size > self.size_column-1 and index >= 0:
                input_batch = self.inputs[:, :, index*effect_sample_size:]
                target_batch = self.targets[index*effect_sample_size]
            if (index+1)*effect_sample_size <= self.size_column-1 and index >= 0:
                input_batch = self.inputs[:, :, index*effect_sample_size:
                                          index*effect_sample_size+self.sample_size]
                target_batch = self.targets[index*effect_sample_size]
            if (index+1)*effect_sample_size <= 0 and index < 0:
                input_batch = self.inputs[:, :, index*effect_sample_size:]
                target_batch = self.targets[index*effect_sample_size]
            
            input_batch = torch.unsqueeze(input_batch, 0)
            target_batch = torch.unsqueeze(target_batch, 0)
            input_batches = torch.cat((input_batches, input_batch), dim=1)
            target_batches = torch.cat((target_batches, target_batch), dim=1)
        
        return input_batches, target_batches 
    
    def __getitem__(self, index=0, overlap=0, rand: bool=False) -> tp.Tuple[torch.Tensor]:
        input_batch, target_batch = self._getCnnItem(index, rand)
        assert overlap >= 0, 'Batch overlap must be non-negative'
        if overlap == 0:
            return input_batch, target_batch
        else:
            target_batch = target_batch[1*overlap:-1*overlap, :]
            # target_batch = target_batch[2*overlap:, :]
            return input_batch, target_batch
        
    def __getnext__(self) -> tp.Tuple[torch.Tensor]: 
        if self.index == self.__len__():
            self.index = 0
        input_batch, target_batch = self.__getitem__(self.index)
        self.index += 1
        return input_batch, target_batch

    def __getitems__(self, rand: bool=False) -> tp.Tuple[torch.Tensor]: 

        input_batches = self.inputs.unfold(2, self.sample_size, 1)
        target_batches = self.targets[:-self.sample_size+1]

        input_batches = torch.permute(input_batches, (2, 0, 1, 3))

        tmp_input = torch.zeros(self.sample_size-1, input_batches.size()[1], input_batches.size()[2], input_batches.size()[3])
        tmp_target = torch.zeros(self.sample_size-1)
        input_batches = torch.cat((input_batches, tmp_input), dim=0)
        target_batches = torch.cat((target_batches, tmp_target), dim=0)

        input_batches = input_batches.unfold(0, self.batch_len, self.batch_len)
        target_batches = target_batches.unfold(0, self.batch_len, self.batch_len)
        input_batches = torch.permute(input_batches, (0, 4, 1, 2, 3))

        target_batches_real = target_batches.real
        target_batches_imag = target_batches.imag
        target_batches_real = torch.unsqueeze(target_batches_real, dim=2)
        target_batches_imag = torch.unsqueeze(target_batches_imag, dim=2)
        target_batches = torch.cat((target_batches_real, target_batches_imag), dim=2)

        return input_batches, target_batches 

    def __getfull__(self, sig_len: int, rand: bool=False) -> tp.Tuple[torch.Tensor]:    
        input = self.inputs.unfold(2, self.sample_size, 1)
        target = self.targets[:-self.sample_size+1]
        input = torch.permute(input, (2, 0, 1, 3))

        tmp_input = torch.zeros(self.sample_size-1, input.size()[1], input.size()[2], input.size()[3])
        tmp_target = torch.zeros(self.sample_size-1)
        input = torch.cat((input, tmp_input), dim=0)
        target = torch.cat((target, tmp_target), dim=0)

        target_real = target.real
        target_imag = target.imag
        target_real = torch.unsqueeze(target_real, dim=1)
        target_imag = torch.unsqueeze(target_imag, dim=1)
        target = torch.cat((target_real, target_imag), dim=1)

        return input, target