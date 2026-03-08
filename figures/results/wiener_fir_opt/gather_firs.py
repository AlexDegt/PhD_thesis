import sys, os

sys.path.append(r"../../lib")

import plot_lib as pl

import matplotlib.pyplot as plt
import numpy as np
import torch
plt.rcParams['font.family'] = 'Times New Roman'

curr_path = os.getcwd()

fir_folder_path = os.path.join(curr_path, "data", "opt_filters", "after_cqa")

fir_union_0dB = np.load(os.path.join(fir_folder_path, "fir_union_0dB.npy"))
fir_union_10dB = np.load(os.path.join(fir_folder_path, "fir_union_10dB.npy"))
fir_union_20dB = np.load(os.path.join(fir_folder_path, "fir_union_20dB.npy"))
fir_union_30dB = np.load(os.path.join(fir_folder_path, "fir_union_30dB.npy"))
fir_union_40dB = np.load(os.path.join(fir_folder_path, "fir_union_40dB.npy"))
fir_union_50dB = np.load(os.path.join(fir_folder_path, "fir_union_50dB.npy"))
fir_union_60dB = np.load(os.path.join(fir_folder_path, "fir_union_60dB.npy"))

fir_unions = [fir_union_0dB, fir_union_10dB, fir_union_20dB, fir_union_30dB, fir_union_40dB, fir_union_50dB, fir_union_60dB]
fir_union_names = ["fir_union_0dB", "fir_union_10dB", "fir_union_20dB", "fir_union_30dB", "fir_union_40dB", "fir_union_50dB", "fir_union_60dB"]

weights_path = os.path.join("data", "wiener_branch_4_taps_11_no_power_ramp_lr_10", "weights_best.pt")

scale_factor = 16384 ** 4

# x_test = np.load(os.path.join(fir_folder_path, "x_test.npy"))
# d_test = np.load(os.path.join(fir_folder_path, "d_test.npy")) / scale_factor
# y_test = np.load(os.path.join(fir_folder_path, "y_test.npy")) / scale_factor

weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

branch_num = 4
filter_num = 3

tap_num = 11

for fir_union, fir_union_name in zip(fir_unions, fir_union_names):
    
    for j_branch in range(branch_num):
        for j_filter in range(filter_num):
            
            curr_fir_imag = fir_union[j_branch * filter_num * tap_num * 2 + j_filter * tap_num * 2 + tap_num * 0: \
                                      j_branch * filter_num * tap_num * 2 + j_filter * tap_num * 2 + tap_num * 1]
            curr_fir_real = fir_union[j_branch * filter_num * tap_num * 2 + j_filter * tap_num * 2 + tap_num * 1: \
                                      j_branch * filter_num * tap_num * 2 + j_filter * tap_num * 2 + tap_num * 2]
            curr_fir = curr_fir_real + 1j * curr_fir_imag
            weights[f"fir_cells.{j_branch}.{j_filter}.weight"] = torch.tensor(curr_fir)
            
    torch.save(weights, os.path.join(fir_folder_path, "weights_" + fir_union_name + ".pt"))
            
    
# weights_check = torch.load(os.path.join(fir_folder_path, "weights_fir_union_30dB.pt"), map_location=torch.device('cpu'), weights_only=True)

fontsize = 13
nfft = 2048
xlim = [2.14 - 0.1, 2.14 + 0.1]
freqs = np.linspace(xlim[0], xlim[1], nfft)

# Compensation before hardware optimization
initial_folder_path = os.path.join(curr_path, "data", "opt_filters", "initial")
x = np.load(os.path.join(initial_folder_path, "x_test.npy"))
y = np.load(os.path.join(initial_folder_path, "y_test.npy")) / scale_factor
d = np.load(os.path.join(initial_folder_path, "d_test.npy")) / scale_factor
noise = 0.009 * (np.random.randn(len(x)) + 1j * np.random.randn(len(x)))
scale = max(abs(x))

colors = [
    '#1F77B4',  # синий
    '#FF7F0E',  # оранжевый
    '#7F7F7F',  # серый
    '#BCBD22',  # желто-зеленый
    '#9467BD',  # фиолетовый
    '#8C564B',  # коричневый
    '#E377C2',  # розовый
    '#2CA02C',  # зеленый
    '#D62728',  # красный
]

pl.plot_psd((x + d) / scale + noise, nfig=1, clf=False, nfft=nfft, freqs=freqs, color=colors[0])
pl.plot_psd((x + d - y) / scale + noise, nfig=1, clf=False, nfft=nfft, freqs=freqs, color=colors[1])

x_no_compens = (x + d - y) / scale + noise

x_compens = []
names = ["_60dB", "_50dB", "_40dB", "_30dB", "_20dB", "_10dB", "_0dB"]
for j, name in enumerate(names):
    x = np.load(os.path.join(fir_folder_path, "x_test" + name + ".npy"))
    y = np.load(os.path.join(fir_folder_path, "y_test" + name + ".npy")) / scale_factor
    d = np.load(os.path.join(fir_folder_path, "d_test" + name + ".npy")) / scale_factor 
    
    x_compens.append((x + d - y) / scale + noise)
    pl.plot_psd((x + d - y) / scale + noise, nfig=1, clf=False, nfft=nfft, freqs=freqs, color=colors[2 + j])
    
plt.legend(["DPD выкл.",
            "DPD вкл., без аппарат. оптимизации/MCM",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-60$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-50$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-40$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-30$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-20$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-10$ дБ",
            r"DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=0$ дБ",], fontsize=13, loc='upper right')    

plt.xlabel("Частота, ГГц", fontsize=15)
plt.ylabel("СПМ, дБм/ГГц", fontsize=15)
plt.ylim([-45, 15])
plt.yticks(np.arange(-45, 15, 10), fontsize=14)
plt.xticks(np.arange(1, 4, 0.025), fontsize=14)
plt.xlim(xlim)


plt.figure(2)
for j in range(len(names)):
# for j in range(3):
    pl.plot_psd(x_compens[j] - x_no_compens, nfig=2, clf=False, nfft=nfft, freqs=freqs, color=colors[2 + j])
    
plt.legend([r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-60$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-50$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-40$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-30$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-20$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=-10$ дБ",
            r"$\Delta$PSD, DPD вкл., CQA/CQA+MCM, max$_i$ $\Delta$ H$_i=0$ дБ",], fontsize=13, loc='upper right')    

plt.xlabel("Частота, ГГц", fontsize=15)
plt.ylabel("СПМ, дБм/ГГц", fontsize=15)
plt.ylim([-160, 90])
plt.yticks(np.arange(-160, 90, 20), fontsize=14)
plt.xticks(np.arange(1, 4, 0.025), fontsize=14)
plt.xlim(xlim)