# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:45:36 2020

@author:
    Sayfullin Karim (swx959511)

@description:
    This library provides some tools to import/export data, compensate delays,
    complex gains, calculating nmse.

"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat as loadmat
#import xlsxwriter as excel

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
        ds_param - array with shape (N,), where N - number
        of signals in dataset described by ds_param
        
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

def papr(x):
    """
        Returns Peak-to-average power
        ratio of input signal x
    """
    l = np.size(x)
    return 10.0*np.log10(l*np.max(np.abs(x)**2)/np.sum(np.abs(x)**2))

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
    return 10.0*np.log10(np.fft.fftshift(psd))

def normalize(x):
    """ Returns normalized array x """
    return x/np.max(np.abs(x))

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
 
def dict2txt(txtname, dictry):
    """
        Saves dictionary dictry to
        .txt file with name txtname
    """
    file = open(txtname, 'w') 
    for key, val in dictry.items():
    	file.write(str(key)+' '+str(val)+'\n')
    file.close()
    return None

def txt2dict(txtname):
    """
        Unpacks dictionary from .txt file
        to dictionary in the script
    """
    dictry = {} 
    file = open(txtname,'r')   
    for line in file.readlines():
        val = []
        line = line.strip()
        key = line.split(' ')[0]
        val_str = np.array(line.split(' ')[1:])
        for token in range(np.size(val_str)):
            f = filter(str.isdecimal, val_str[token])
            val.append(int("".join(f)))
        val_size = np.size(val)
        if val_size == 1:
            dictry[key] = val[0]
        else:
            dictry[key] = val
    file.close()
    return dictry

def create_expname(ds_param, sig_len, pr, aver, pj_distr):
    """
        Function creates and returns name of experiment
        with current experiment parameters
    """
    sig_num = ds_param.shape[1]
    nonrep_types1 = analyze_ds(ds_param[0, :])[0].astype(int)
    nonrep_types2 = analyze_ds(ds_param[0, :])[0].astype(int)
    return None

def prepare_environm(ds_param, sig_len, pr, aver, pj_distr):
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
    return None

#def write_nmse_to_excel(*data, datapath='../feic/Results.xlsx'):
#    """ Write data to excel table"""
#
#    workbook = excel.Workbook(datapath)
#    worksheet = workbook.add_worksheet()
#    row = 0
#    col = 0
#
#    for dat in data:
#        col = 0
#        for num in dat:
#            worksheet.write(col, row, num)
#            col += 1
#        row += 1
#
#    workbook.close()
#
#
#def write2d_to_excel(data, datapath='../feic/Results.xlsx'):
#    """ Write data to excel table"""
#
#    M, N = data.shape
#    workbook = excel.Workbook(datapath)
#    worksheet = workbook.add_worksheet()
#    row = 0
#    col = 0
#
#    for ii in range(M):
#        col = 0
#        for jj in range(N):
#            worksheet.write(row, col, data[ii][jj])
#            col += 1
#        row += 1
#
#    workbook.close()
