import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.lines import Line2D
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat
# import torch

from scipy.integrate import simps
from scipy.integrate import trapz

plt.rcParams["font.family"] = "Times New Roman"

def aclr_fn(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
    """ 
        Calculate Adjacent Channel Leakage Ratio
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    ind1 = (nfft // 2) - int(np.ceil(nfft * f/fs))
    ind2 = (nfft // 2) + int(np.ceil(nfft * f/fs))
    guard = 4 # 10
    aclr = np.sum(psd[ind2 + guard: 2 * ind2 - ind1 + guard])/np.sum(psd[ind1: ind2])
    return 10 * np.log10(aclr)

def get_psd(sig, Fs=1.0, nfft=2048, window='blackman', nperseg=None, noverlap=None, scale='log'):
    """ 
        Returns Power Spectral Density of input signal sig
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    if scale == 'log':
        return freqs, 10*np.log10(np.fft.fftshift(psd))
    elif scale == 'lin':
        return freqs, np.fft.fftshift(psd)
    else:
        raise ValueError(f"scale parameter must equal \'log\' or \'lin\', but {scale} is given.")

def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=None, is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, 
             noverlap=None, y_shift=0, color=None, fontsize=13, xlabel='frequency', ylabel='Magnitude [dB]', scale='log'):
    """ Plotting power spectral density """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()
        
    if isinstance(color, list):
        assert len(signals) == len(color) or (len(color) == 1)
        for c in color:
            assert isinstance(c, str)
        if len(color) == 1:
            color *= len(signals)
    elif isinstance(color, str) or color is None:
        color = [color] * len(signals)
    else:
        raise TypeError(f"color parameter must be either of a str or list of str type, but {type(color)} is given")
      
    ax.set_xlabel(xlabel, fontsize=fontsize)
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)

    for j_sig, iisignal in enumerate(signals):
        # freqs = np.linspace(-Fs/2, Fs/2, iisignal.size)
        # plt.plot(freqs, 10*np.log10(np.fft.fftshift(np.fft.fft(iisignal))))

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, 1, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)*Fs
        freqs += xshift
        if scale == 'log':
            psd = 10*np.log10(np.fft.fftshift(psd)) + y_shift
        elif scale == 'lin':
            psd =  np.fft.fftshift(psd) + y_shift
        else:
            raise ValueError(f"scale parameter must equal \'log\' or \'lin\', but {scale} is given.")
        ax_ptr, = ax.plot(freqs, psd, color=color[j_sig])
#        ax_ptr, = ax.plot(freqs, psd, color='tab:blue')

    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if legend is not None:
        ax.legend(legend, fontsize=fontsize)
    if is_save:
        nfig.savefig(filename)
    plt.show()
    return ax_ptr

# Determine powers for model input for train and test correspondingly
out_power_raugh = np.array([-11.6, -10.5, -9.5, -8.5, -7.6, -6.8, -6, -5.2, -4.5, -3.8, -3.1, -2.5, -1.9, -1.3, -0.8, -0.4])
in_power_ref = 10 ** (np.arange(-16, 0, 1) / 10)
out_power_ref = 10 ** ((30 + out_power_raugh) / 10)
in_power = list(10 ** ((np.load('pa_powers_round.npy') - 1)/ 10))
# out_powers in W
pa_powers = np.interp(in_power, in_power_ref, out_power_ref) / 1000

power_cases_train = pa_powers[::2]
power_cases_test = pa_powers[1::2]

# Determine parameters of the simulation, which is chosen for PSD and ACLR(pa_power) graphs:
pow_param_num = 10
param_num = 22
delay_num = 4
slot_num = 4

f = 10 / 1000 # GHz
fs = 245.76 / 1000 # GHz
nfft = 512

figure = 1
ylim = [-50, -10]

# add_to_name = ""
# add_to_name = "_ls"
# add_to_name = "_1e_3"
# add_to_name = "_1e_4"
add_to_name = "_3e_6"

# tx_test = np.load(os.path.join(f'x_test{add_to_name}.npy'))
# pa_out_test = np.load(os.path.join(f'd_test{add_to_name}.npy'))
# model_out_test = np.load(os.path.join(f'y_test{add_to_name}.npy'))

tx_test = np.load(os.path.join(f'x_test{add_to_name}.npy'))
pa_out_test = np.load(os.path.join(f'd_test{add_to_name}.npy'))
model_out_test = np.load(os.path.join(f'y_test{add_to_name}.npy'))

sig_len_test = int(len(tx_test) / len(power_cases_test))

aclr_train = []
aclr_test = []

psd_train, psd_test, psd = [], [], []

for i in range(len(power_cases_test)):
    x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
    d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    y = model_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    scale = np.max(abs(x))
    x /= scale
    d /= scale
    y /= scale
    aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
    aclr_test.append(aclr_val)
    print(f"Case test {power_cases_test[i]:.2f} W: ACLR = {aclr_val} dB")
    
plt.figure(figure)
# if j_folder == 0:
#     plt.plot(pa_powers, aclr_one_dim_sep, marker='o')
# plt.plot(pa_powers[::2], aclr_train, marker='o')
plt.plot(pa_powers[1::2], aclr_test, marker='o')
plt.xlabel('PA output power, W', fontsize=13)
plt.ylabel('ACLR, dB', fontsize=13)
plt.ylim(ylim)
plt.yticks(np.arange(15, -60, -5))

# Concatenate arrays to write in .txt file
# folder_name = "aclr_text"
# aclr_correct_train_tmp, aclr_correct_test_tmp, aclr_correct_tmp = [], [], []
# aclr = np.zeros((len(aclr_train) + len(aclr_test),)).tolist()
# aclr[::2] = aclr_train
# aclr[1::2] = aclr_test

# aclr_correct_train_tmp.append(pa_powers[::2])
# aclr_correct_test_tmp.append(pa_powers[1::2])
# aclr_correct_tmp.append(pa_powers)

# aclr_correct_train_tmp.append(aclr_train)
# aclr_correct_test_tmp.append(aclr_test)
# aclr_correct_tmp.append(aclr)

# aclr_correct_train_tmp = np.array(aclr_correct_train_tmp).T
# aclr_correct_test_tmp = np.array(aclr_correct_test_tmp).T
# aclr_correct_tmp = np.array(aclr_correct_tmp).T

# Save arrays to .txt file to draw picture in latex
# np.savetxt(os.path.join(os.getcwd(), folder_name, f'aclr_correct_dim{add}_train.txt'), aclr_correct_train_tmp)
# np.savetxt(os.path.join(os.getcwd(), folder_name, f'aclr_correct_dim{add}_test.txt'), aclr_correct_test_tmp)
# np.savetxt(os.path.join(os.getcwd(), folder_name, f'aclr_correct_dim{add}.txt'), aclr_correct_tmp)