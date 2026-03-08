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

x = loadmat("new_data_07_01_2024/ofdm_2.5M_qam16_qpsk_fs_163p84_256k")["s"][0, :]
d = np.load("new_data_07_01_2024/2024_01_07_ofdm_2.5M_qam16_qpsk_RF814MHz_LO859MHz_Pm30dbm_800mks_10GSa_PIERS_fs163p84MHz_FIRatt20dB.npy")

nfft = 2048
f0 = 814
fs = 163.84
xlim = [f0 - fs / 2, f0 + fs / 2]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(x, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, МГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/МГц", fontsize=fontsize)
plt.ylim([-75, 0])
plt.xlim(xlim)

xlim = [0 - 81.73 + 859, 0 + 81.73 + 859]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(d, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, МГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/МГц", fontsize=fontsize)
plt.ylim([-75, 0])
plt.xlim(xlim)


