# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:23:35 2021

@author: dWX1065688
"""

import plot_lib as pl
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat as loadmat
import numpy as np

def fir_matrix_generate(x, M):
    """
        Generating matrix for FIR-filter with order M and definite model centre
        x is supposed to be row
    """
    N = np.size(x)
    U = np.zeros((N+M-1, M), dtype = 'complex128')
    for i in range(M):
        U[i:i+N, i] = x.reshape((-1,))
    cut_part = np.floor(M/2).astype(int)
    U = U[cut_part:N+M-1-cut_part, :]
    return U
    
def fir_filtering_matrix(P, w):
    """
        Function filters columns of matrix P,
        filter with impulse response w
        Function uses state matrix of each column
    """
    M = np.size(w)
    w = w.reshape((M,))
    M = np.size(w)
    P_shape = P.shape; col_num = P_shape[1]
    P_filtered = np.zeros(P_shape, dtype = 'complex128')
    for i in range(col_num):
        U_curr = fir_matrix_generate(P[:, i].T, M)
        P_filtered[:, i] = U_curr @ w
    return P_filtered

def fir_filtering_matrix_conv(P, w):
    """
        Function filters columns of matrix P,
        filter with impulse response w
        Function convolves each column
    """
    M = np.size(w)
    w = w.reshape((M,))
    P_shape = P.shape; col_num = P_shape[1]
    P_filtered = np.zeros(P_shape, dtype = 'complex128')
    for i in range(col_num):
        P_filtered[:, i] = signal.convolve(P[:, i], w, mode='same')
    return P_filtered

def hammerstein_forward(M, w_lut, w_fir, V):
    z = V @ w_lut
    U = fir_matrix_generate(z, M)
    y = U @ w_fir
    return y, z

def delay_compensate(x, d):
    """
        Delay compensation between input and desired signals
    """
    data_corr_abs = np.abs(signal.correlate(x**2, d))
    max_elem = np.max(data_corr_abs)
    index_max = data_corr_abs.tolist().index(max_elem)
    corr_half_index = int((np.size(data_corr_abs) + 1)/2)
    delta_index = index_max - corr_half_index
#    print(delta_index)
    d = np.roll(d, delta_index)
    return x, d

def lut_matrix_generate(x, nspl, pwr = 2):
    """
        Function returns 1D LUT statement matrix
    """
    N = np.size(x)
    V = np.zeros((N, nspl), 'complex128')
    left_spline = np.zeros(N, int)
    right_spline = np.zeros(N, int)
    x_lut = x*nspl
    for i in range(N):
        tx_floor = np.floor(np.abs(x_lut[i]))
        left_spline[i] = int(tx_floor - 0)
        right_spline[i] = left_spline[i] + 1
        delta = (np.abs(x_lut[i]) - tx_floor)
        V[i][left_spline[i]] = (x[i]**pwr)*(1 - delta)
        V[i][right_spline[i]] = (x[i]**pwr)*delta
    return V

def analyze_nmse(nmse, nmse_newton, sym_num):
    """
        Returns 3 arrays of defferences.
        nmse - NMSE calculates with certain algorithm on sum_num signals
        nmse_newton - NMSE calculated with Newton algorithm on sum_num signals
        sym_num - number of signals in dataset
        Function calculates deviation input signal nmse from nmse_newton and
        returns nmse values at the beginning of each signal, 
        at the middle and at the end.
    """
    sym_len = int(len(nmse)/sym_num)
    nmse_peak = np.zeros((sym_num,))
    nmse_centre = np.zeros((sym_num,))
    nmse_end = np.zeros((sym_num,))
    nmse_peak_newton = np.zeros((sym_num,))
    nmse_centre_newton = np.zeros((sym_num,))
    nmse_end_newton = np.zeros((sym_num,))
    for i in range(sym_num):
        nmse_peak[i] = nmse[i*sym_len]
        nmse_centre[i] = nmse[i*sym_len + int(np.floor(sym_len/2))]
        nmse_end[i] = nmse[(i + 1)*sym_len - 1]
        nmse_peak_newton[i] = nmse_newton[i*sym_len]
        nmse_centre_newton[i] = nmse_peak_newton[i]
        nmse_end_newton[i] = nmse_centre_newton[i]
    plt.figure(1)
    plt.title("NMSE peak values")
    plt.plot(nmse_peak - nmse_peak_newton)
    plt.figure(2)
    plt.title("NMSE centre values")
    plt.plot(nmse_centre - nmse_centre_newton)
    plt.figure(3)
    plt.title("NMSE end values")
    plt.plot(nmse_end - nmse_end_newton)
    return nmse_peak_newton - nmse_peak, nmse_centre_newton - nmse_centre, nmse_end_newton - nmse_end

def analyze_ds(ds_param):
    """
        Detects all types of signal in dataset, 
        described in 2-dim array ds_param.
        Counts frquencies of occurance of each single signal.
        ds_param has length ds_len
        
        Function returns array of NON-REPEATED types of signal
        and occurance frequencies the signal types
    """
    nonrep = []
    freq = []
    ds_len = ds_param.size
    for i in range(ds_len):
        if not (ds_param[i] in nonrep):
            nonrep.append(ds_param[i])
            freq.append(1)
        else:
            freq[nonrep.index(ds_param[i])] += 1
    tmp = zip(nonrep, freq)
    tmp_ = sorted(tmp, key=lambda tup: tup[0])
    nonrep = [tmp[0] for tmp in tmp_]
    freq = [tmp[1] for tmp in tmp_]
    
    types_info = np.vstack([np.array(nonrep), np.array(freq)])
    return types_info

def pad_zeros(matr, MAX_ROW, toarr = True, trps = True):
    """
        Adds zeros so that each row of input 
        list matr has MAX_ROW of them.
        toarr - flag that indicates whether to 
        transform list to array or not
        toarr = True - transform list to array
        toarr = False - don`t transform list to array
    """
    row_num = np.shape(matr)[0]
    for i in range(row_num):
        curr_row_nonz_len = len(matr[i])
        matr[i].extend([0 for j in range(MAX_ROW-curr_row_nonz_len)])
    if trps == True:
        return np.array(matr).T
    else:
        return np.array(matr)

def import_data(filename, datatype):
    """ Import data with chosen datatype and printing it """
    if datatype == 'mat':
        res = loadmat(filename)
        # for key, value in res.items():
        #     print(key, value)
        return res
    elif datatype == 'np':
        return np.load(filename)
    elif datatype == 'txt_cmp':
        # Read cmpx data stored in txt file (FEIC)
        data_init = np.loadtxt(filename, dtype='int32')
        N = len(data_init)
        data_real = np.zeros(int(N/2))
        data_imag = np.zeros(int(N/2))

        for ii in range(0, N, 2):
            data_real[int(ii/2)] = data_init[ii]
            data_imag[int(ii/2)] = data_init[ii+1]

        data = np.complex64(data_real + data_imag*1j)
        return data
    else:
        print('Error in datatype')
        return None

def nmse(x, e):
    """ Returns Normalized Mean Squared error """
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e))/np.sum(x*np.conj(x)))))
    return y

def nmse_rt(x, e):
    """ Returns Normalized Mean Squared error calculated in real time """
    N = x.size
    L = e.size
    y = 10.0*np.log10(np.real(((N*np.sum(e*np.conj(e)))/(L*np.sum(x*np.conj(x))))))
    return y

def mse(e):
    """ Returns Mean Squared error """
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e)))))
    return y

def normalize(x):
    return x/np.max(np.abs(x))

def papr(x):
    l = np.size(x)
    return 10.0*np.log10(l*np.max(np.abs(x)**2)/np.sum(np.abs(x)**2))

def dataset_generate(ds_len, sig_len, path_len, pa_pow_len):
    ds_param = np.zeros((3, ds_len), dtype = int)
    ds_param[0, :] = (np.floor(sig_len*np.random.rand(ds_len))).astype(int)
    ds_param[1, :] = (np.floor(path_len*np.random.rand(ds_len))).astype(int)
    ds_param[2, :] = (np.floor(pa_pow_len*np.random.rand(ds_len))).astype(int)
    return ds_param

def max_ind(z):
    len_z = len(z)
    if len_z <= 0:
        print('Array must have at least 1 element')
        return 0
    max_elem = z[0]
    ind = 0
    for i in range(len(z)):
        if z[i] > max_elem:
            max_elem = z[i]
            ind = i
    return ind

def get_psd(sig, Fs=1.0, nfft=2048, window='blackman', nperseg=None, noverlap=None):
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    return 10.0*np.log10(np.fft.fftshift(psd))