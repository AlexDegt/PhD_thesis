import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat
# import torch

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

def get_psd(sig, Fs=1.0, nfft=2048, window='blackman', nperseg=None, noverlap=None):
    """ 
        Returns Power Spectral Density of input signal sig
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    return freqs, 10*np.log10(np.fft.fftshift(psd))

def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=None, is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, 
             noverlap=None, y_shift=0, color=None, fontsize=13, xlabel='frequency', ylabel='Magnitude [dB]'):
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
        psd = 10.0*np.log10(np.fft.fftshift(psd)) + y_shift
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

dim = 10
figure = 2

f = 10 # MHz
fs = 245.76 # MHz
nfft = 1024

tx_train = np.load('x.npy')
pa_out_train = np.load('d.npy')
model_out_train = np.load('y.npy')
tx_test = np.load('x_test.npy')
pa_out_test = np.load('d_test.npy')
model_out_test = np.load('y_test.npy')

pa_powers = list(10 ** ((np.load('pa_powers_round.npy') - 1)/ 10))
power_cases_train = pa_powers[::2]
power_cases_test = pa_powers[1::2]


sig_len_train = int(len(tx_train) / len(power_cases_train))
sig_len_test = int(len(tx_test) / len(power_cases_test))

aclr_train = []
aclr_test = []

psd_train, psd_test, psd_train_nc, psd_test_nc, psd, psd_nc = [], [], [], [], [], []

for i in range(len(power_cases_train)):
    x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
    d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    y = model_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    scale = np.max(abs(x))
    x /= scale
    d /= scale
    y /= scale
    aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
    aclr_train.append(aclr_val)
    freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
    # plot_psd(x + d - y, f=f, fs=fs, nfft=nfft)
    # sys.exit()
    psd_train.append(psd)
    print(f"Case train {power_cases_train[i]:.2f} dB: ACLR = {aclr_val} dB")
    # if i == 0:
    #     plt.figure(i + 1)
    #     plt.title(f"Power case {power_cases_train[i]} dB")
    #     pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    # plt.figure(i + 1)
    # plt.title(f"Power case {power_cases_train[i]} dB")
    # pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    
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
    freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
    psd_test.append(psd)
    print(f"Case test {power_cases_test[i]:.2f} dB: ACLR = {aclr_val} dB")
    # plt.figure(i + 1)
    # plt.title(f"Power case {power_cases_test[i]} dB")
    # pl.plot_psd(x + d, x + d - y, nfig=i + 1, clf=False)
    
psd = list(np.zeros((len(pa_powers),)))
psd[::2] = psd_train
psd[1::2] = psd_test
psd = np.array(psd)
    
aclr_train_no_correct, aclr_test_no_correct = [], []
# Calculate ACLR for PA output without correction
for i in range(len(power_cases_train)):
    x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
    d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
    scale = np.max(abs(x))
    x /= scale
    d /= scale
    # plot_psd(x + d, f=f, fs=fs, nfft=nfft)
    # sys.exit()
    aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
    aclr_train_no_correct.append(aclr_val)
    freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
    psd_train_nc.append(psd_val)
    
for i in range(len(power_cases_test)):
    x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
    d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
    scale = np.max(abs(x))
    x /= scale
    d /= scale
    aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
    aclr_test_no_correct.append(aclr_val)
    freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
    psd_test_nc.append(psd_val)
    
psd_nc = list(np.zeros((len(pa_powers),)))
psd_nc[::2] = psd_train_nc
psd_nc[1::2] = psd_test_nc
psd_nc = np.array(psd_nc)

aclr_no_correct = list(np.zeros((len(pa_powers),)))
aclr_no_correct[::2] = aclr_train_no_correct
aclr_no_correct[1::2] = aclr_test_no_correct

plt.figure(figure)
plt.plot(pa_powers[::2], [p for p in aclr_train], marker='o')
plt.plot(pa_powers[1::2], [p for p in aclr_test], marker='o')
# plt.plot(pa_powers, [p for p in aclr_no_correct], marker='o', color='black')
plt.xlabel('power, mW', fontsize=13)
plt.ylabel('ACLR, dB', fontsize=13)
# plt.yticks(np.arange(10, 50, 5))
plt.legend(['train', 'test'], fontsize=13)
plt.grid()

# psd_min = -75
# psd_max = -21
# psd[psd < psd_min] = psd_min
# # psd[psd >= -46] = -46
# psd_nc[psd_nc < psd_min] = psd_min
# # psd_nc[psd_nc >= -46] = -46

# # Draw PSD color map
# pa_range = np.arange(psd_min - 1, psd_max, 1)
# levels = list(pa_range)
# plt.figure(figure * 100)
# F, P = np.meshgrid(freqs, pa_powers)
# cs = plt.contourf(F, P, psd, levels=levels, cmap ="jet")
# cbar = plt.colorbar(cs) 
# plt.ylabel("PA power", fontsize=13)
# plt.xlabel("freq, MHz", fontsize=13)

# levels = list(pa_range)
# plt.figure(figure * 200)
# F, P = np.meshgrid(freqs, pa_powers)
# cs = plt.contourf(F, P, psd_nc, levels=levels, cmap ="jet")
# cbar = plt.colorbar(cs) 
# plt.ylabel("PA power", fontsize=13)
# plt.xlabel("freq, MHz", fontsize=13)