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
pa_out_us /= 1.3

# psd = sl.get_psd(sig_together)
x_psd = sl.get_psd(x_us)
pa_out_psd = sl.get_psd(pa_out_us)

f_left = 1.885
f_right = 1.975
step = 0.01

x_ax = np.linspace(f_left, f_right, len(x_psd))
plt.plot(x_ax, x_psd, color='blue')
plt.plot(x_ax, pa_out_psd, color='red')
# plt.plot(x_ax, psd, color='blue')
# plt.xticks(np.arange(f_left, f_right + step, step))
plt.ylim([-60, 15])
plt.xlim([f_left + (step / 2), f_right - (step / 2)])
plt.legend(["Сигнал на входу УМ", "Сигнал на выходе УМ"], fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.grid()
