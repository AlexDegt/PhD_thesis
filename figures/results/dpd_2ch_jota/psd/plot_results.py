import os, sys
sys.path.append('../../../lib')
import plot_lib as pl
import support_lib as sl
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # Добавлен импорт
import scipy.signal as signal

plt.rcParams["font.family"] = "Times New Roman"

fontsize = 13

curr_path = os.getcwd()

rvcnn_folder = "paper_exp_4_seed_968_newton_lev_marq_4_channels_6_5_5_2_ker_size_3_3_3_3_act_sigmoid_1500_epochs"
cvcnn_folder = "reproduced_paper_exp_2_seed_966_complex_start_simple_cubic_newton_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_5000_epochs"

rvcnn_path = os.path.join(curr_path, rvcnn_folder)
cvcnn_path = os.path.join(curr_path, cvcnn_folder)

x = np.load(os.path.join(rvcnn_path, "x.npy"))
d = np.load(os.path.join(rvcnn_path, "d.npy"))
y_rvcnn = np.load(os.path.join(rvcnn_path, "y.npy"))
y_cvcnn = np.load(os.path.join(cvcnn_path, "y.npy"))

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

x_non_comp = signal.convolve(x + d, taps)
x_dpd_rvcnn = signal.convolve(x + d - y_rvcnn, taps)
x_dpd_cvcnn = signal.convolve(x + d - y_cvcnn, taps)

noise_x = 0.002 * (np.random.randn(len(x_non_comp)) + 1j * np.random.randn(len(x_non_comp)))
# noise_x = 0.009 * (np.random.randn(len(x_non_comp)) + 1j * np.random.randn(len(x_non_comp)))

x_non_comp += noise_x
x_dpd_rvcnn += noise_x
x_dpd_cvcnn += noise_x

nfft = 2048
xlim = [1.99 - 0.4, 1.99 + 0.4]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_psd(x_non_comp, x_dpd_rvcnn, x_dpd_cvcnn, freqs=freqs, nfft=nfft)
plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("СПМ, дБм/ГГц", fontsize=fontsize)
plt.ylim([-55, 20])
plt.yticks(np.arange(-60, 40, 10))
plt.xticks(np.arange(1, 4, 0.05))
plt.xlim([1.75, 2.2])

plt.legend(["TX канал A, DPD выкл.",
            "TX канал A, DPD вкл., RV-CNN, NM-LM",
            "TX канал A, DPD вкл., CV-CNN, CMNM"], fontsize=13)