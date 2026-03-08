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

mat = loadmat("data2d")

xA = mat["PDinA"][0, :]
xB = mat["PDinB"][0, :]
pa_outA = mat["PDoutA"][0, :]
pa_outB = mat["PDoutB"][0, :]

xA /= max(abs(xA))
xB /= max(abs(xB))
pa_outA /= max(abs(pa_outA))
pa_outB /= max(abs(pa_outB))

# Параметры фильтра
order = 201           # Порядок фильтра (длина фильтра = order + 1)
fs = 1           # Частота дискретизации, Гц
cutoff = 0.3       # Частота среза, Гц
transition_width = 0.05  # Ширина переходной полосы, Гц

# Границы полос для алгоритма Ремеза
bands = [0, cutoff - transition_width/2,     # Полоса пропускания
         cutoff + transition_width/2, fs/2]  # Полоса задерживания

# Желаемые амплитуды в полосах [0, cutoff] и [cutoff, fs/2]
desired = [1, 0]

# Веса ошибок в полосах (можно регулировать для оптимизации)
weights = [1, 1]

# Проектирование фильтра методом Ремеза
taps = signal.remez(order + 1, bands, desired, weight=weights, fs=fs)
# taps = np.array([0, 1, 0])

# pl.plot_firfr(taps)

xA = signal.convolve(xA, taps)
xB = signal.convolve(xB, taps)
pa_outA = signal.convolve(pa_outA, taps)
pa_outB = signal.convolve(pa_outB, taps)

xA_ = signal.resample(xA, 2 * len(xA))
xB_ = signal.resample(xB, 2 * len(xB))
pa_outA_ = signal.resample(pa_outA, 2 * len(pa_outA))
pa_outB_ = signal.resample(pa_outB, 2 * len(pa_outB))

xA_ = xA_ * np.exp(-2j * np.pi * np.arange(len(xA_)) * 150 / 800)
xB_ = xB_ * np.exp(-2j * np.pi * np.arange(len(xB_)) * (-150) / 800)
pa_outA_ = pa_outA_ * np.exp(-2j * np.pi * np.arange(len(pa_outA_)) * 150 / 800)
pa_outB_ = pa_outB_ * np.exp(-2j * np.pi * np.arange(len(pa_outB_)) * (-150) / 800)

noise_x = 0.0003 * (np.random.randn(len(xA_)) + 1j * np.random.randn(len(xA_)))
noise_pa_out = 0.009 * (np.random.randn(len(pa_outA_)) + 1j * np.random.randn(len(pa_outA_)))

x = xA_ + xB_ + noise_x
pa_out = pa_outA_ + pa_outB_ + noise_pa_out

nfft = 2048
xlim = [1.99 - 0.4, 1.99 + 0.4]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(x, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-70, 23])
plt.yticks(np.arange(-70, 40, 10))
plt.xticks(np.arange(1, 4, 0.05))
plt.xlim(xlim)

pl.plot_psd(pa_out, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-40, 23])
plt.yticks(np.arange(-70, 40, 10))
plt.xticks(np.arange(1, 4, 0.05))
plt.xlim(xlim)

noise_err = 0.0045 * (np.random.randn(len(pa_outA)) + 1j * np.random.randn(len(pa_outA)))

errA = pa_outA - xA + noise_err
errB = pa_outB - xB + noise_err

nfft = 2048
xlim = [0 -200, 0 + 200]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(errA, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, МГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/МГц", fontsize=fontsize)
plt.ylim([-60, 23])
plt.xlim(xlim)

pl.plot_psd(errB, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, МГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/МГц", fontsize=fontsize)
plt.ylim([-60, 10])
plt.xlim(xlim)