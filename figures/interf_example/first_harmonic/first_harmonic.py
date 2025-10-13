import os, sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.signal as signal

sys.path.append(r'D:\Projects\MIPT\PhD thesis\Tex\figures\lib')

plt.rcParams['font.family'] = 'Times New Roman'

import plot_lib as pl
import support_lib as sl
import numpy as np

mat = loadmat("aligned_m16_00dB_100RB_Fs245p76")

fontsize = 13

x = mat["TX"].reshape(-1)
pa_out = mat["PAout"].reshape(-1)

sig_len = len(x)

x_us = x
pa_out_us = pa_out
x_us = signal.resample(x, int(sig_len / 2))
pa_out_us = signal.resample(pa_out, int(sig_len / 2))

x_us /= max(abs(x_us))
pa_out_us /= max(abs(pa_out_us))
scale = np.sqrt(10 ** (30 / 10))
pa_out_us /= scale

fs = 245.76e6   # частота дискретизации
f_shift_1 = 40e6  # сдвиг частоты
f_shift_2 = 42e6  # сдвиг частоты
n = np.arange(len(x_us))

# Сдвиг частоты
x_us = x_us * np.exp(-1j * 2 * np.pi * f_shift_1 * n / fs)
pa_out_us = pa_out_us * np.exp(1j * 2 * np.pi * f_shift_2 * n / fs)

sig_together = x_us + pa_out_us

# psd = sl.get_psd(sig_together)
x_psd = sl.get_psd(x_us)
pa_out_psd = sl.get_psd(pa_out_us)

x_ax = np.linspace(1.90, 1.99, len(x_psd))
plt.plot(x_ax, x_psd, color='blue')
plt.plot(x_ax, pa_out_psd, color='red')
# plt.plot(x_ax, psd, color='blue')
plt.xticks(np.arange(1.90, 1.99 + 0.01, 0.01))
plt.ylim([-70, 5])
plt.xlim([1.91, 1.985])
plt.legend(["Сигнал передатчика, TX", "Помеха на приёмнике, RX"], fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.grid()
