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
import alex_degt_support_lib as sup
import feic_lib as feic
import numpy as np
import math

''' Constants '''
START_wangshw_21_11_23 = feic.START_wangshw_21_11_23
ALPHA_wangshw_21_11_23 = np.array([0.022, 0.024, 0.036, 0.043, 0.050, 0.061, 0.065, 0.078, 0.091, 0.098, 0.113, 0.151, 0.170, 0.195, 0.237, 0.261, 0.272, 0.286, 0.292, 0.300])*7
IMP_RESP_wangshw_21_11_23 = np.load(r'../../Useful_data/wangshiwei_23_11_21/multipath/imp_resp_set.npy')
VGA_pwr_aver_npr_22_02_07 = feic.VGA_pwr_aver_npr_22_02_07.astype(str)
RB_wangshw_21_11_23 = feic.RB_wangshw_21_11_23.astype(str)
DATA_PR_122K_SAMP_NONAVER  = feic.DATA_PR_122K_SAMP_NONAVER 
DATA_NOPR_122K_SAMP_NONAVER  = feic.DATA_NOPR_122K_SAMP_NONAVER
DATA_PR_122K_SAMP_AVER  = feic.DATA_PR_122K_SAMP_AVER
DATA_NOPR_122K_SAMP_AVER = feic.DATA_NOPR_122K_SAMP_AVER 

""" Preprocessing """
vga_gain = VGA_pwr_aver_npr_22_02_07
start_names = START_wangshw_21_11_23
RB_names = RB_wangshw_21_11_23
alpha = ALPHA_wangshw_21_11_23
imp_resp = IMP_RESP_wangshw_21_11_23

def gen_ds_RBeqPW(ds_param, exp_name, N, pr, aver, pj=None):
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
    '''
    if N == 122880 and pr == True and aver == True:
        data_folder = DATA_PR_122K_SAMP_AVER
    elif N == 122880 and pr == False and aver == True:
        data_folder = DATA_NOPR_122K_SAMP_AVER
    elif N == 122880 and pr == True and aver == False:
        data_folder = DATA_PR_122K_SAMP_NONAVER
    elif N == 122880 and pr == False and aver == False:
        data_folder = DATA_NOPR_122K_SAMP_NONAVER
    try:
        os.mkdir(r'../data/'+data_folder)
    except:
        pass
    RB_ind = feic.RB2ind(ds_param[0, :])
    ds_len = ds_param.shape[1]
    M = imp_resp.shape[0]
    order = M-1
    prefix = np.zeros((order,), dtype='complex128')
    curr_sig = np.zeros((N+order,), dtype='complex128')
    ''' Define array of phase jumps '''
    if np.size(pj) == 1:
        if pj == None:
            pj = np.ones(ds_len)
    ''' Create directories to save LUT and FIR coefficients '''
    os.mkdir('../data/'+data_folder+'/'+exp_name)
    os.mkdir('../data/'+data_folder+'/'+exp_name+'/tx')
    os.mkdir('../data/'+data_folder+'/'+exp_name+'/rx')
    for i in range(ds_len):
        mat_file = sup.import_data(r'../../../../../feic_big_data/'+data_folder+\
        '/LTE20M_'+RB_names[RB_ind[i]]+'RB_Start'+start_names[RB_ind[i]]+'_16QAM_122k_VGA_gain_'+vga_gain[RB_ind[i]]+'_iter_1.mat', 'mat')
        pa_out = mat_file['y'].reshape((-1,))
        tx = mat_file['pdin'].reshape((-1,))
        tx /= np.max(np.abs(tx))
        pa_out /= np.max(np.abs(pa_out))
        pa_out *= alpha[RB_ind[i]]
        pa_out *= np.exp(1j*pj[i])
        curr_sig[0:order] = prefix.copy()
        curr_sig[order:N+order] = pa_out.copy()
        prefix = pa_out[N-order:N].copy()
        np.savez_compressed('../data/'+data_folder+'/'+exp_name+'/tx/'+str(i), tx)
        rx = np.convolve(curr_sig, (imp_resp[:, ds_param[1, i]]/np.max(np.abs(np.fft.fftshift(np.fft.fft(imp_resp[:, 5], 1024))))).reshape((-1,)), mode='valid')
        np.savez_compressed('../data/'+data_folder+'/'+exp_name+'/rx/'+str(i), rx)
        if i % 500 == 0:
            print('Signal %i is already defined' %i)
    return None