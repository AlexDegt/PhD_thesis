# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:45:36 2020

@author:
    Sayfullin Karim (swx959511)

@description:
    This library provides some tools to import/export data, compensate delays,
    complex gains, calculating nmse.

"""

import os
import sys
import re
import numbers
import numpy as np
import argparse
import feic_lib as feic
import scipy.signal as signal
import matplotlib.pyplot as plt
import dataset_generator as dsg
from scipy.io import loadmat as loadmat
from time import perf_counter

def fir_matrix_generate(x, M):
    """
        Generating matrix for FIR-filter with order M and definite model centre
        x is supposed to be row
    """
    N = np.size(x)
    U = np.zeros((N+M-1, M), dtype = 'complex128')
    for i in range(M):
        # if len(x.shape) == 2:
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

def grad_est(w_lut, w_fir, x, d):
    """
        Estimate gradient,
        Works for block algorithms
    """
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = lut_matrix_generate(x, nspl, pwr = 2)
    V_f = fir_filtering_matrix_conv(V, w_fir)
    z = V @ w_lut
    U = fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U))
    V_full_H = np.conj(V_full).T
    e = d - (U @ w_fir)
    return (V_full_H @ e).reshape((-1))

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
        ds_param - array with shape (N,), where N - number
        of signals in dataset described by ds_param
        
        Function returns array of NON-REPEATED types of signal
        and occurance frequencies the signal types
    """
    nonrep = []
    freq = []
    ds_len = np.size(ds_param)
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

def power_ramp(x, leng, window='base', start=True, finish=True):
    """  """
    y = np.copy(x)
    if window == 'base':
        win = np.linspace(0, 1, leng)
    elif window == 'no':
        return y
    
    if start:
        y[:leng] = win * y[:leng]
    if finish:
        y[-leng:] = np.flip(win) * y[-leng:]
    return y
    
def plotter(ax, datax, datay, param_dict):
    " Plot helper "
    if len(datax) == 1:
        datax = np.arange(len(datay))
    out = ax.plot(datax, datay, **param_dict)
    return out

def compensate_delay2(x, y):
    """ Conpensate delay difference between two signals. This function cut out 
    correlated parts of the signal. Full synchronization is not guaranteed. """
    corr = np.abs(signal.correlate(x, y))
    maxi = np.argmax(corr)
    center = int((len(corr)+1)/2)-1

    c_low = 0
    c_high = len(corr)
    y_low = maxi - int((len(y)/2))
    y_high = maxi + int((len(y)/2))
    x_low = center - int((len(x)/2))
    x_high = center + int((len(x)/2))
    low = np.max((c_low, y_low, x_low))
    high = np.min((c_high, y_high, x_high))

    dyl = low - y_low
    dyh = high - y_low
    y_out = y[dyl:dyh]
    dxl = low - x_low
    dxh = high - x_low
    x_out = x[dxl:dxh]
    return x_out, y_out

def compensate_delay(x, y, d=0, is_write=False):
    """ Compensating delay difference between two signals.
        Requires cmp gain compensation before
        If d != 0, then:
            d>0 means that x forward y
            d<0 means that x backward y
        """
    if d == 0:  # (We don't know delay)
        conv = signal.correlate(x, y)
        leng = len(conv) + 1
        maxi = np.argmax(conv)
        delay = int(leng/2 - maxi - 1)
    else:
        delay = d

    if delay > 0:
        x = x[:-delay]
        y = y[delay:]
        if is_write:
            print('There is some delay =', delay, 'compensated!')
    elif delay < 0:
        y = y[:-int(-1*delay)]
        x = x[int(-1*delay):]
        if is_write:
            print('There is delay below zero:', delay, 'compensated!')
    else:
        if is_write:
            print('There is no delay to compensate. Delay is', delay)
    return x, y

def compensate_cmpx_gain(x, d, method='LS', Ntr=16):
    """ Calculate (LS or RLS) complex gain to minimize error between x and d
    and return compensated signal """

    if method == 'LS':
        if np.ndim(x) == 1:
            x.reshape(len(x), 1)
        if np.ndim(d) == 1:
            d.reshape(len(d), 1)
        U = x
        UH = np.transpose(np.conjugate(x))
        wd = UH.dot(d)/UH.dot(U)
        y = wd*x
    elif method == 'WRLS':
        lam = 0.999
        llam = (1/lam)
        P = 0.001
        N = len(x)
        w = 1+0j
        y = np.zeros((N,), dtype='complex128')
        for nn in range(N):
            y[nn] = w*x[nn]
            e = d[nn] - y[nn]
            k = (P*np.conj(x[nn])) / (lam + ((x[nn]*P) * np.conj(x[nn])))
            w = w + k*e
            P = llam*(P - ((k*x[nn]) * P))
    elif method == 'LS_batch':
        N = len(x)
        if N % 2:
            print('Error, input must be odd')
            return None
        winlen = int(np.ceil(N/(Ntr+1))) * 2
        window = signal.windows.triang(winlen).reshape((winlen, 1))
        window = window/np.max(window)
        ndop = int(((winlen/2)*(Ntr+1) - N))
        while ndop % 2:
            Ntr -= 1
            winlen = int(np.ceil(N/(Ntr+1))) * 2
            window = signal.windows.triang(winlen).reshape((winlen, 1))
            ndop = int(((winlen/2)*(Ntr+1) - N))
        zer = np.zeros((int(ndop/2),), dtype='complex128')
        x = np.hstack((zer, x, zer))
        d = np.hstack((zer, d, zer))
        Ntr += 2
        zeroes_dop = np.zeros((int(winlen/2)), dtype='complex128')
        x = np.hstack((zeroes_dop, x, zeroes_dop))
        d = np.hstack((zeroes_dop, d, zeroes_dop))

        y = np.zeros((len(x), 1), dtype='complex128')
        ind = 0

        for ii in range(Ntr):
            xtmp = x[ind:ind+winlen].reshape((winlen, 1)) * window
            dtmp = d[ind:ind+winlen].reshape((winlen, 1)) * window
            UH = np.transpose(np.conjugate(xtmp))
            wd = UH.dot(dtmp)/UH.dot(xtmp)
            ytmp = wd*xtmp
            y[ind:ind+winlen, :] = y[ind:ind+winlen, :] + ytmp
            ind += int(winlen/2)

        y = y[int(winlen/2)+int(ndop/2):-int(winlen/2)-int(ndop/2)]
        y = y.reshape((len(y),))
    else:
        y = -1
        print('Wrong type of algorithm')
    return y

def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2
 # See if left child of root exists and is
 # greater than root 
    if l < n and arr[i] < arr[l]:
        largest = l 
 # See if right child of root exists and is
 # greater than root 
    if r < n and arr[largest] < arr[r]:
        largest = r 
 # Change root, if needed 
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap 
  # Heapify the root. 
        heapify(arr, n, largest)
  
# The main function to sort an array of given size
def heapSort(arr):
    n = len(arr) 
 # Build a maxheap.
 # Since last parent will be at ((n//2)-1) we can start at that location. 
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
 # One by one extract elements
    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0)
 

def partition(array, start, end):
    """
        Supporting function for quick sort
    """
    pivot = array[start]
    low = start + 1
    high = end
    while True:
        # If the current value we're looking at is larger than the pivot
        # it's in the right place (right side of pivot) and we can move left,
        # to the next element.
        # We also need to make sure we haven't surpassed the low pointer, since that
        # indicates we have already moved all the elements to their correct side of the pivot
        while low <= high and array[high] >= pivot:
            high = high - 1
        # Opposite process of the one above
        while low <= high and array[low] <= pivot:
            low = low + 1
        # We either found a value for both high and low that is out of order
        # or low is higher than high, in which case we exit the loop
        if low <= high:
            array[low], array[high] = array[high], array[low]
            # The loop continues
        else:
            # We exit out of the loop
            break
    array[start], array[high] = array[high], array[start]
    return high

def qsort(array, start, end):
    """
        Quick sort
    """
    if start >= end:
        return
    p = partition(array, start, end)
    qsort(array, start, p-1)
    qsort(array, p+1, end)

def search_eff_int(x, perct=0.9, sort='h'):
    """
    Function searches for effective interval (x_min, x_max)
    Effective interval includes perct (0 <= perct <= 1) of points in the array x
    Elements of x must be float numbers

    Returns
    -------
    x_min, x_max, x_mean, perct_result
    
    """
    t1 = perf_counter()
    x_copy = x.copy()
    len_x = len(x_copy)
    if sort == 'h':
        heapSort(x_copy)
    if sort == 'q':
        qsort(x_copy, 0, len_x - 1)  
    x_mean = np.mean(x_copy)
    ind_high = np.where(((x_copy <= np.max(x_copy)) == (x_mean < x_copy)) == True)[0]    
    ind_low = np.arange(ind_high[0])
    # print(ind_high)
    # print(ind_low)
    first_inc_ind = int(np.ceil(len(ind_low)*(1 - perct)))    
    last_inc_ind = len_x - 1 - int(np.ceil(len(ind_high)*(1 - perct))) 
    # print(first_inc_ind)
    # print(last_inc_ind)
    x_min = x_copy[first_inc_ind]
    x_max = x_copy[last_inc_ind]
    # print(x_min, x_mean, x_max)
    perct_result = (last_inc_ind - first_inc_ind + 1)/len_x
    t2 = perf_counter()
    # print('Elapsed time: {:.5} sec'.format(t2-t1))
    return x_min, x_max, x_mean, perct_result

def nmse(x, e):
    """ Returns Normalized Mean Squared error """
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e))/np.sum(x*np.conj(x)))))
    return y

def nmse_nf(x, e, nf):
    """ Returns Normalized Mean Squared error """
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e))-np.linalg.norm(nf)**2)/(np.sum(x*np.conj(x))-np.linalg.norm(nf)**2)))
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

def papr(x):
    """
        Returns Peak-to-average power
        ratio of input signal x
    """
    l = np.size(x)
    return 10.0*np.log10(l*np.max(np.abs(x)**2)/np.sum(np.abs(x)**2))

def ber(input_bits, output_bits):
    """ Calculate Bit Error Rate """
    return sum(input_bits != output_bits)/len(input_bits)

def SNR(signal, noise):
    """ Calculate signal to noise ratio """
    return 10*np.log10(np.linalg.norm(signal)/np.linalg.norm(noise))

def max_ind(z):
    """
        Returns index of the biggest element 
        of the array z.
        Works only with real arrays
    """
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
    """ 
        Returns Power Spectral Density of input signal sig
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    return 10*np.log10(np.fft.fftshift(psd))

def normalize(x, y):
    """ Returns normalized array x """
    return x/np.max(np.abs(y))

def aclr(x, e, fx, fe, bw, Fs, ntaps=512):
    """ Returns Adjascent channel leakage ratio """
    w0 = bw*0.5 / Fs
    h = signal.firwin(ntaps+1, w0)
    t = np.linspace(0, ntaps, ntaps+1) - ntaps/2.0
    hx = h * np.exp(1j * 2 * np.pi * fx/Fs * t)
    he = h * np.exp(1j * 2 * np.pi * fe/Fs * t)
    xfilt = signal.lfilter(hx, 1, x)
    efilt = signal.lfilter(he, 1, e)
    return nmse(xfilt, efilt)

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

def pause():
    input('Press any key to continue...')
 
def dict2txt(txtname, dictry, mode='w'):
    """
        Saves dictionary dictry to
        .txt file with name txtname
    """
    file = open(txtname, mode) 
    for key, val in dictry.items():
        file.write(str(key)+':\n')
        if (isinstance(val, numbers.Number) or isinstance(val, str)):
            file.write(str(val)+'\n')
        else:
            size_val = np.shape(val)[0]
            for i in range(size_val):
                file.write(str(val[i])+'\n')
    file.close()
    return None

def txt2dict(txtname, content_type):
    """
        Unpacks dictionary from .txt file
        to dictionary in the script.
        content_type defines expected type
        of data within the dictionary values.
        content type could be:
        FLOAT, INT, STR
    """
    dictry = {} 
    file = open(txtname,'r')   
    for line in file.readlines():
        line = line.strip()
        if line[-1] == ':':
            key = line[:-1]
            val = []
        else:
            if content_type == float:
                val.append(str2float(line))
            elif content_type == str:
                val.append(line)
            elif content_type == int:
                val.append(str2int(line))
        dictry[key] = val
    file.close()
    for key, val in dictry.items():
        if np.size(dictry[key]) == 1:
            dictry[key] = val[0]
    return dictry

def dict2arr(dictry):
    """
        Function transforms dictionaries to 
        array type objects
    """  
    return np.array(list(dictry.values())[0]).T

def str2float(string):
    """
        Function have string as an input
        and converts it to list of float numbers
    """
    list_float = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", string)
    list_float = np.array(list_float).astype(float).tolist()
    list_float_size = np.size(list_float)
    if list_float_size == 1:
        return list_float[0]
    else:
        return list_float
    
def str2int(string):
    """
        Function have string as an input
        and converts it to list of int numbers
    """
    float_nums = str2float(string)
    if (isinstance(float_nums, numbers.Number)):
        return int(float_nums)
    else:
       return np.array(float_nums).astype(int).tolist() 

def create_short_dataname(ds_param, gen_ds_param_dictry):
    """
        Function creates and returns short experiment name
        with current datatset parameters
    """
    sig_len = gen_ds_param_dictry[feic.gen_ds_keys.SIGNAL_LENGTH.value]
    pr = gen_ds_param_dictry[feic.gen_ds_keys.POWER_RAMP.value]
    aver = gen_ds_param_dictry[feic.gen_ds_keys.DATA_AVERAGE.value]
    pj = gen_ds_param_dictry[feic.gen_ds_keys.PHASE_JUMP_DISTR.value]
    pj_distr = gen_ds_param_dictry[feic.gen_ds_keys.PHASE_JUMP_DISTR_PARAM.value]
    sig_num = np.shape(ds_param)[1]
    nonrep_types1 = analyze_ds(ds_param[0])[0].astype(int)
    nonrep_types2 = analyze_ds(ds_param[1])[0].astype(int)
    nonrep_type1_size = np.size(nonrep_types1)
    nonrep_type2_size = np.size(nonrep_types2)
    short_name = str(sig_num)+' sigs; '
    short_name += str(sig_len)+' samples; '
    short_name += pr+'; '+aver+'; '
    if pj == 'None':
        short_name += 'None '
    else:
        short_name += pj+' '
    if pj_distr != '[]':
        short_name += str(pj_distr)+' '
    short_name += 'pj; '  
    for i in range(nonrep_type1_size):
        short_name += str(nonrep_types1[i])+' '
    short_name += 'RB; '
    for i in range(nonrep_type2_size):
        short_name += str(nonrep_types2[i])+' '
    short_name += 'path; '
    return short_name

def create_short_expname(ds_param, gen_ds_param_dictry, alg_param_dictry, model_param_dictry, sym_param_dictry):
    """
        Function creates and returns short experiment name
        with current datatset parameters
    """
    alg_keys = list(alg_param_dictry.keys())
    model_keys = list(model_param_dictry.keys())
    sym_keys = list(sym_param_dictry.keys())
    short_name = 'Alg param: '
    for i in range(np.size(alg_keys)):
        short_name += str(alg_param_dictry[alg_keys[i]])+'_'+alg_keys[i]+'; '
    short_name += '\nModel param: '
    for i in range(np.size(model_keys)):
        short_name += str(model_param_dictry[model_keys[i]])+'_'+model_keys[i]+'; '
    short_name += '\nSym param: '
    for i in range(np.size(sym_keys)):
        short_name += str(sym_param_dictry[sym_keys[i]])+'_'+sym_keys[i]+'; '
    short_name += '\nDataset param: '
    short_name += create_short_dataname(ds_param, gen_ds_param_dictry)
    return short_name

def short_name_file_add(ds_param, gen_ds_param_dictry, alg_param_dictry, model_param_dictry, sym_param_dictry, data_or_exp, folder=''):
    """
        Function adds short description of 
        DATASET with signals ds_param and
        parameters gen_ds_param_dictry
        to the file with short names of datasets
        dataset_description.txt in the "folder"
    """
    if data_or_exp == 'data':
        name = create_short_dataname(ds_param, gen_ds_param_dictry)
        description_name = '\dataset_description.txt'
    elif data_or_exp == 'exp':
        name = create_short_expname(ds_param, gen_ds_param_dictry, alg_param_dictry, model_param_dictry, sym_param_dictry)
        description_name = '\experiment_description.txt'
    ds_descr_folder = folder+'\\'
    content = os.listdir(ds_descr_folder)
    content_size = np.size(content)
    decimals = []
    is_decimal = []
    for i in range(content_size):
        is_decimal.append(content[i].isdecimal())
        if is_decimal[i] == True:
            decimals.append(int(content[i]))
    if decimals == []:
        data_folder_name = str(0)
    else:
        data_folder_name = str(np.max(decimals)+1)
    dict2txt(ds_descr_folder+description_name, {data_folder_name: name}, 'a')
    return None

def prepare_environm(ds_param, alg_param, model_param, gen_ds_param, phase_jumps, save_param, sym_param, alg_folder):
    """
        Function provides name of experiment, name of folder with data
        for model learning according to experiment parameters.
        Also it creates .txt file with all experiment parameters and 
        datatset(!) according to this parameters. It is created in the same folder.
        If there is a file with such parameters, file with parameters
        and dataset won`t be created.
        
        ds_param - array 2xN parameters of signals (N signals), for example
        RB numbers in 0-s row and mulipathes names in 1-st row
        sig_len - number of samples in 1 signal (typical - 122880)
        pr - flag that indicates if there are power ramps
        aver - flag that indicates if there is an averaging
        pj_distr - list that consist of:
            1) name of power jumps distribution, for example 'norm'
            2) parameters of distribution, for example:
            mu (expectation), sigma (standart deviation) for normal distribution
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))+'\\'
    exp_fold_abs_path = ''
    dataset_fold_abs_path = ''
    ''' Prepare ds_param for comparison '''
    ds_param_suit = np.zeros((np.shape(ds_param)[1], 2), dtype=int)
    ds_param_suit[:, 0] = ds_param[0]
    ds_param_suit[:, 1] = ds_param[1]
    ds_param_suit = ds_param_suit.tolist()
    ds_param_suit = [str(ds_param_suit[i]) for i in range(np.shape(ds_param_suit)[0])]
    ''' Prepare alg_param for comparison '''
    alg_param_suit = np.array(alg_param).T.tolist()
    ''' Prepare model_param for comparison '''
    model_param_suit = np.array(model_param).T.tolist()
    ''' Prepare gen_ds_param for comparison '''
    gen_ds_param_suit = np.array(gen_ds_param).T.tolist()
    ''' Prepare phase_jumps for comparison '''
    phase_jumps_suit = np.array(phase_jumps).T.astype(str).tolist()
    ''' Prepare save_param for analyzis'''
    save_param_suit = np.array(save_param).T.tolist()
    ''' Prepare sym_param for analyzis'''
    sym_param_suit = np.array(sym_param).T.tolist()
    
    ds_param_dictry = {'RB_and_path': ds_param_suit}
    phase_jumps_dictry = {'Distribution': phase_jumps_suit}
    alg_param_dictry = dict(zip(alg_param_suit[0], alg_param_suit[1]))
    model_param_dictry = dict(zip(model_param_suit[0], model_param_suit[1]))
    gen_ds_param_dictry = dict(zip(gen_ds_param_suit[0], gen_ds_param_suit[1]))
    save_param_dictry = dict(zip(save_param_suit[0], save_param_suit[1]))
    sym_param_dictry = dict(zip(sym_param_suit[0], sym_param_suit[1]))
    
    ''' Check whether there is appropriate DATASET that conforms input parameters. If not - create new '''
    ds_param_match_fold = search_exp(ds_param_dictry, 'ds_param.txt', reg_exp=r'[-+]?\d+', folder=curr_dir+r'..\dataset\data', part_match=True)
    gen_ds_param_match_fold = search_exp(gen_ds_param_dictry, 'gen_ds_param.txt', reg_exp=r'[-+]?\d+', folder=curr_dir+r'..\dataset\data', part_match=False)
    phase_jumps_match_fold = search_exp(phase_jumps_dictry, 'phase_jumps.txt', reg_exp=r'[-+]?\d+', folder=curr_dir+r'..\dataset\data', part_match=False)
    
    dataset_folder_list = list(set(ds_param_match_fold) & set(gen_ds_param_match_fold) & set(phase_jumps_match_fold))
    is_appr_dataset = ([] != dataset_folder_list)
    if is_appr_dataset:
        dataset_folder = dataset_folder_list[0]
    else:
        short_name_file_add(ds_param, gen_ds_param_dictry, {}, {}, {}, \
                                'data', folder=curr_dir+r'..\dataset\data')
        dataset_folder = add_exp_folder(curr_dir+r'..\dataset\data')
        dict2txt(curr_dir+'..\dataset\data\\'+dataset_folder+'\\ds_param.txt', ds_param_dictry)
        dict2txt(curr_dir+r'..\dataset\data\\'+dataset_folder+'\\gen_ds_param.txt', gen_ds_param_dictry)
        dict2txt(curr_dir+r'..\dataset\data\\'+dataset_folder+'\\phase_jumps.txt', phase_jumps_dictry)
        if sym_param_dictry['-gsim'] == 'True':
            dsg.gen_ds_RBeqPW_GSIM(ds_param, phase_jumps, gen_ds_param_dictry, model_param_dictry, folder=r'..\dataset\data\\'+dataset_folder)
        else:
            dsg.gen_ds_RBeqPW(ds_param, phase_jumps, gen_ds_param_dictry, model_param_dictry, folder=r'..\dataset\data\\'+dataset_folder)
    dataset_fold_abs_path = curr_dir+'..\dataset\data\\'+dataset_folder
        
    ''' Check wheter there is appropriate EXPERIMENT that conforms input parameters. If not - create new '''
    alg_param_match_fold = search_exp(alg_param_dictry, 'alg_param.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    model_param_match_fold = search_exp(model_param_dictry, 'model_param.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    ds_param_match_fold = search_exp(ds_param_dictry, 'ds_param.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    gen_ds_param_match_fold = search_exp(gen_ds_param_dictry, 'gen_ds_param.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    phase_jumps_match_fold = search_exp(phase_jumps_dictry, 'phase_jumps.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    sym_match_fold = search_exp(sym_param_dictry, 'sym_param.txt', reg_exp=r'[-+]?\d+', folder=alg_folder+r'results', part_match=False)
    
    ''' Experiment technique parameters '''
    save_res = save_param_dictry['-svr']
    
    experiment_folder_list = list(set(alg_param_match_fold) & set(model_param_match_fold) & set(sym_match_fold) & \
                                  set(ds_param_match_fold) & set(gen_ds_param_match_fold) & set(phase_jumps_match_fold))
    is_appr_experiment = ([] != experiment_folder_list)

    if is_appr_experiment:
        exp_folder = experiment_folder_list[0]
        print('Results of experiment with such dataset and parameters already exist. It is in folder %s\n' %(experiment_folder_list[0]))
        sys.exit(1)
    else:
        if save_res == 'True':
            short_name_file_add(ds_param, gen_ds_param_dictry, alg_param_dictry, model_param_dictry, sym_param_dictry, 'exp', folder=alg_folder+r'results')
            exp_folder = add_exp_folder(alg_folder+r'results')
            exp_fold_abs_path = alg_folder+r'results\\'+exp_folder
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\alg_param.txt', alg_param_dictry)
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\model_param.txt', model_param_dictry)
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\ds_param.txt', ds_param_dictry)
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\gen_ds_param.txt', gen_ds_param_dictry)
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\phase_jumps.txt', phase_jumps_dictry)
            dict2txt(alg_folder+r'results\\'+exp_folder+'\\sym_param.txt', sym_param_dictry)
            ''' Create directories to save LUT and FIR coefficients '''
            os.mkdir(alg_folder+r'results\\'+exp_folder+r'\w_lut')
            os.mkdir(alg_folder+r'results\\'+exp_folder+r'\w_fir')
            os.mkdir(alg_folder+r'results\\'+exp_folder+r'\perform')
            os.mkdir(alg_folder+r'results\\'+exp_folder+r'\state')
            np.save(alg_folder+r'results\\'+exp_folder+'\\ds_param.npy', ds_param)

    return dataset_fold_abs_path, exp_fold_abs_path

def search_exp(dictry, txt_name, reg_exp=r'[-+]?\d+', folder='', part_match=False):
    """
        Function searches for .txt files
        which has the same content as the input dictionary.
        The name of directory to search in is an absolute path which is an input parameter "folder"
        The names of folders to search among is defined by 
        regular expression reg_exp.
        Function returns list of names folders which include .txt 
        file with the same content as in the inputdictionary.
    """
    folder = folder+'\\'
    content_dig = []
    content_all = os.listdir(folder)
    match_folders = []
    for i in range(np.size(content_all)):
        match = re.match(reg_exp, content_all[i])
        if match: content_dig.append(match[0])
        
    for fold in range(np.size(content_dig)):
        for curdict in range(np.size(dictry)):
            dictry_txt = txt2dict(folder+content_dig[fold]+'\\'+txt_name, str)
            # if txt_name == 'ds_param.txt':
            #     print(list(dictry_txt.values()))
            if np.size(list(dictry_txt.values())) == 1 and type(list(dictry.values())[0]) == list:
                dictry_txt.update({list(dictry_txt.keys())[0]: list(dictry_txt.values())})
            ''' Debug code '''
            # if txt_name == 'ds_param.txt':
            #     print(dictry_txt)
                # print(dictry)
            if not part_match:
                if dictry == dictry_txt:
                    match_folders.append(content_dig[fold])        
            else:
                dictry_values = list(dictry.values())
                dictry_txt_values = list(dictry_txt.values())
                dictry_size = np.size(dictry_values)
                dictry_txt_size = np.size(dictry_txt_values)
                if (dictry.keys() == dictry_txt.keys()) and \
                (dictry_values[0][0:dictry_size] == dictry_txt_values[0][0:dictry_size]) and \
                (dictry_size <= dictry_txt_size):
                    match_folders.append(content_dig[fold]) 
    return match_folders

def add_exp_folder(folder=''):
    """
        Adds experiment folder.
        Function analyzes current folders content,
        Finds folder called by serial number and
        adds folder called by the next serial number
        If there is no folders called with just serial numbers, 
        function adds folder called '0'
    """
    folder = folder+'\\'
    content = os.listdir(folder)
    content_size = np.size(content)
    decimals = []
    is_decimal = []
    for i in range(content_size):
        is_decimal.append(content[i].isdecimal())
        if is_decimal[i] == True:
            decimals.append(int(content[i]))
    if decimals == []:
        os.mkdir(folder+str(0))
        return str(0)
    else:
        os.mkdir(folder+str(np.max(decimals)+1))
        return str(np.max(decimals)+1)

def createParser(alg_param, argv):
    alg_param_size = np.shape(alg_param)[0]
    parser = argparse.ArgumentParser()
    for i in range(alg_param_size):
        parser.add_argument(alg_param[i][0], default=alg_param[i][1])
    namespace = parser.parse_args(argv)
    return namespace

def findElemListDictry(dictry_list, elem_str):
    '''
        Finds index of key in dictionary
        with list type
    '''
    return np.array(dictry_list).T[0].tolist().index(elem_str)