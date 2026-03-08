# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:23:35 2021

@author: dWX1065688
"""

import os
import sys
import plot_lib as pl
import scipy.signal as signal
import matplotlib.pyplot as plt
import support_lib as sl
import feic_lib as feic
import numpy as np

''' Constants '''
curr_dir = os.path.dirname(os.path.realpath(__file__))+'/'
START_wangshw_21_11_23 = feic.START_wangshw_21_11_23
ALPHA_wangshw_21_11_23 = np.array([0.022, 0.024, 0.036, 0.043, 0.050, 0.061, 0.065, 0.078, 0.091, 0.098, 0.113, 0.151, 0.170, 0.195, 0.237, 0.261, 0.272, 0.286, 0.292, 0.3])*2.5
# ALPHA_22_06_01 = np.ones((20,), dtype=int)
try:
    IMP_RESP_wangshw_21_11_23 = np.load(curr_dir+r'../Useful_data/wangshiwei_23_11_21/multipath/imp_resp_set.npy')
except:
    IMP_RESP_wangshw_21_11_23 = np.zeros((48, 6), dtype=complex)
    
# IMP_RESP_wangshw_21_11_23 = np.load(curr_dir+r'../Useful_data/wangshiwei_23_11_21/multipath/imp_resp_set_4master_thesis.npy')
VGA_pwr_aver_npr_22_02_07 = feic.VGA_pwr_aver_npr_22_02_07.astype(str)
# VGA_pwr_aver_npr_22_02_07 = feic.VGA_pwr_aver_pr_22_05_31.astype(str)
RB_wangshw_21_11_23 = feic.RB_wangshw_21_11_23.astype(str)

""" Preprocessing """
vga_gain = VGA_pwr_aver_npr_22_02_07
start_names = START_wangshw_21_11_23
RB_names = RB_wangshw_21_11_23
alpha = ALPHA_wangshw_21_11_23
# alpha = ALPHA_22_06_01
imp_resp = IMP_RESP_wangshw_21_11_23

def gen_ds_RBeqPW(ds_param, phase_jumps, gen_ds_param_dictry, model_param_dictry, folder=''):
    '''
        Generate dataset for the case when RB number
        unambiguously corresponds to power of PA
        (RBeqPW = RB number "equals" power)
        Parameters:
        ds_param - 2 dimesional array. 0-s row: RB number (=power), 1-st row: multipath
        N - length of each signal in dataset
        pr - flag True/False, that shows whether there are power ramps or not
        aver - flag True/False, that shows whether there is an averaging or not
        distr_pj - distribution of phase jumps. For example 'norm' - normal
        
        Version of dataset generator for Python algorithms implementation
    '''
    curr_dir = os.path.dirname(os.path.realpath(__file__))+'\\'
    folder = curr_dir + folder+'\\'
    ds_param = np.array(ds_param)
    phase_jumps = np.array(phase_jumps)
    
    sig_len_str = gen_ds_param_dictry[feic.gen_ds_keys.SIGNAL_LENGTH.value]
    sig_len = sl.str2int(sig_len_str)
    mu = gen_ds_param_dictry[feic.gen_ds_keys.AWGN_EXPECT.value]
    sigma = gen_ds_param_dictry[feic.gen_ds_keys.AWGN_DEVIAT.value]
    if (mu != 'None') and (sigma != 'None'):
        mu = sl.str2float(mu)
        sigma = sl.str2float(sigma)
        noise = np.random.normal(mu, sigma, sig_len)+1j*np.random.normal(mu, sigma, sig_len)
    else:
        noise = 0
    pr = gen_ds_param_dictry[feic.gen_ds_keys.POWER_RAMP.value]
    aver = gen_ds_param_dictry[feic.gen_ds_keys.DATA_AVERAGE.value]
    M = np.size(imp_resp[:, 0])
    order = M - 1
    sig_num = np.shape(ds_param)[1]   
    RB_ind = feic.RB2ind(ds_param[0, :])
    if np.shape(ds_param)[1] == 1:
        RB_ind = RB_ind.reshape((1))
    
    prefix = np.zeros((order,), dtype='complex128')
    curr_sig = np.zeros((sig_len+order,), dtype='complex128')
    ''' Create directories to save LUT and FIR coefficients '''
    os.mkdir(folder+r'tx')
    os.mkdir(folder+r'rx')
    folder_init_data = '\\'+sig_len_str+'_'+pr+'_'+aver
    for i in range(sig_num):
        mat_file = sl.import_data(curr_dir+r'..\..\..\..\..\feic_big_data'+folder_init_data+\
        '/LTE20M_'+RB_names[RB_ind[i]]+'RB_Start'+start_names[RB_ind[i]]+'_16QAM_122k_VGA_gain_'+vga_gain[RB_ind[i]]+'_iter_1.mat', 'mat')
        pa_out = mat_file['y'].reshape((-1,))
        tx = mat_file['pdin'].reshape((-1,))
        tx /= np.max(np.abs(tx))
        pa_out /= np.max(np.abs(pa_out))
        pa_out *= alpha[RB_ind[i]]
        pa_out *= np.exp(1j*phase_jumps[i])
        curr_sig[0:order] = prefix.copy()
        curr_sig[order:sig_len+order] = pa_out.copy()
        prefix = pa_out[sig_len-order:sig_len].copy()
        rx = np.convolve(curr_sig, (imp_resp[:, ds_param[1, i]]/np.max(np.abs(np.fft.fftshift(np.fft.fft(imp_resp[:, 5], 1024))))).reshape((-1,)), mode='valid')
        rx /= np.max(np.abs(tx))
        rx += noise
        np.savez_compressed(folder+r'tx\\'+str(i), tx)
        np.savez_compressed(folder+r'rx\\'+str(i), rx)
        if i % 500 == 0 and i != 0:
            print('Signal %i is already defined' %i)
        if i == 0:
            print('Dataset creation started')
    return None

def gen_ds_RBeqPW_GSIM(ds_param, phase_jumps, gen_ds_param_dictry, model_param_dictry, folder=''):
    '''
        Generate dataset for the case when RB number
        unambiguously corresponds to power of PA
        (RBeqPW = RB number "equals" power)
        Parameters:
        ds_param - 2 dimesional array. 0-s row: RB number (=power), 1-st row: multipath
        N - length of each signal in dataset
        pr - flag True/False, that shows whether there are power ramps or not
        aver - flag True/False, that shows whether there is an averaging or not
        distr_pj - distribution of phase jumps. For example 'norm' - normal
        
        Version of dataset generator for GraphSim algorithms implementation
    '''
    curr_dir = os.path.dirname(os.path.realpath(__file__))+'\\'
    folder = curr_dir + folder+'\\'
    ds_param = np.array(ds_param)
    phase_jumps = np.array(phase_jumps)
    
    sig_len_str = gen_ds_param_dictry[feic.gen_ds_keys.SIGNAL_LENGTH.value]
    sig_len = sl.str2int(sig_len_str)
    mu = gen_ds_param_dictry[feic.gen_ds_keys.AWGN_EXPECT.value]
    sigma = gen_ds_param_dictry[feic.gen_ds_keys.AWGN_DEVIAT.value]
    if (mu != 'None') and (sigma != 'None'):
        mu = sl.str2float(mu)
        sigma = sl.str2float(sigma)
        noise = np.random.normal(mu, sigma, sig_len)+1j*np.random.normal(mu, sigma, sig_len)
    else:
        noise = 0
    pr = gen_ds_param_dictry[feic.gen_ds_keys.POWER_RAMP.value]
    aver = gen_ds_param_dictry[feic.gen_ds_keys.DATA_AVERAGE.value]
    M = np.size(imp_resp[:, 0])
    order = M - 1
    delay_fir = int(np.floor((M)/2))
    sig_num = np.shape(ds_param)[1]   
    RB_ind = feic.RB2ind(ds_param[0, :])
    if np.shape(ds_param)[1] == 1:
        RB_ind = RB_ind.reshape((1))
    
    if M % 2 == 0:
        prefix_len = delay_fir-1
    else:
        prefix_len = delay_fir
    postfix_len = delay_fir
    prefix = np.zeros((prefix_len,), dtype='complex128')
    postfix = np.zeros((postfix_len,), dtype='complex128')
    
    curr_sig = np.zeros((sig_len+order,), dtype='complex128')
    ''' Create directories to save LUT and FIR coefficients '''
    os.mkdir(folder+r'tx')
    os.mkdir(folder+r'rx')
    folder_init_data = '\\'+sig_len_str+'_'+pr+'_'+aver
    for i in range(sig_num):
        mat_file = sl.import_data(curr_dir+r'..\..\..\..\..\feic_big_data'+folder_init_data+\
        '/LTE20M_'+RB_names[RB_ind[i]]+'RB_Start'+start_names[RB_ind[i]]+'_16QAM_122k_VGA_gain_'+vga_gain[RB_ind[i]]+'_iter_1.mat', 'mat')
        pa_out = mat_file['y'].reshape((-1,))
        tx = mat_file['pdin'].reshape((-1,))
        tx /= np.max(np.abs(tx))
        pa_out /= np.max(np.abs(pa_out))
        pa_out *= alpha[RB_ind[i]]
        pa_out *= np.exp(1j*phase_jumps[i])
        
        if i+1 != sig_num:
            mat_file_next = sl.import_data(curr_dir+r'..\..\..\..\..\feic_big_data'+folder_init_data+\
            '/LTE20M_'+RB_names[RB_ind[i+1]]+'RB_Start'+start_names[RB_ind[i+1]]+'_16QAM_122k_VGA_gain_'+vga_gain[RB_ind[i+1]]+'_iter_1.mat', 'mat')
            pa_out_next = mat_file_next['y'].reshape((-1,))
            pa_out_next /= np.max(np.abs(pa_out_next))
            pa_out_next *= alpha[RB_ind[i+1]]
            pa_out_next *= np.exp(1j*phase_jumps[i+1])
            postfix = pa_out_next[0:postfix_len]
        else:
            postfix = np.zeros((postfix_len,), dtype='complex128')
        
        curr_sig[0:prefix_len] = prefix.copy()
        curr_sig[prefix_len:sig_len+prefix_len] = pa_out.copy()
        curr_sig[sig_len+prefix_len:sig_len+order] = postfix.copy()       
        prefix = pa_out[sig_len-prefix_len:sig_len].copy()        
        
        rx = np.convolve(curr_sig, (imp_resp[:, ds_param[1, i]]/np.max(np.abs(np.fft.fftshift(np.fft.fft(imp_resp[:, 5], 1024))))).reshape((-1,)), mode='valid')
        rx /= np.max(np.abs(tx))
        rx += noise
        np.savez_compressed(folder+r'tx\\'+str(i), tx)
        np.savez_compressed(folder+r'rx\\'+str(i), rx)
        if i % 500 == 0 and i != 0:
            print('Signal %i is already defined' %i)
        if i == 0:
            print('Dataset creation started')
    return None