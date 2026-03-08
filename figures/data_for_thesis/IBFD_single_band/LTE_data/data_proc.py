import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

from scipy.linalg import toeplitz, solve

import copy

sys.path.insert(0, "../../lib")
import plot_lib as pl
import support_lib as sl
import scipy.signal as signal
import scipy
from scipy.io import loadmat

fontsize = 13

def deconvolution_frequency(y, w, regularization=1e-10):
    """
    Восстановление x в частотной области
    
    Parameters:
    y : array-like, выходной сигнал
    w : array-like, импульсная характеристика
    regularization : float, параметр регуляризации
    
    Returns:
    x_est : восстановленный сигнал
    """
    # Определяем длину для БПФ
    N_fft = len(y) + len(w) - 1
    
    # Переходим в частотную область
    Y = np.fft.fft(y, N_fft)
    W = np.fft.fft(w, N_fft)
    
    # Обращение фильтра с регуляризацией
    W_inv = np.conj(W) / (np.abs(W)**2 + regularization)
    
    # Восстановление сигнала
    X_est = Y * W_inv
    x_est = np.fft.ifft(X_est)
    
    # Обрезаем до нужной длины
    x_est = x_est[:len(y) - len(w) + 1]
    
    return x_est

mat = loadmat("LTE60M_334RB_fs122k")

w = np.load("../create_filter/channel_filter.npy")

x = mat["TXa"][0, :]
d = mat["RXa"][0, :]

pa_out = deconvolution_frequency(d, w)

# pl.plot_psd(x, pa_out, d)

nfft = 2048
xlim = [1.8 - 0.06144, 1.8 + 0.06144]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(x, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-180, 5])
plt.xlim(xlim)

pl.plot_psd(pa_out, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-75, 0])
plt.xlim(xlim)

pl.plot_psd(d, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-75, 0])
plt.xlim(xlim)

