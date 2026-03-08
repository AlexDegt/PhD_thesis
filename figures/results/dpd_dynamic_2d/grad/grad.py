import sys, os

sys.path.append(f"../../../lib")
import support_lib as sl
import plot_lib as pl

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

fontsize = 13

lc_bgd_2d = np.load("lc_qcrit_test_bgd_12_pow_dim_inp_stand_m1_1_50_epochs_mu_1e_4.npy")
# lc_1d = np.load("lc_qcrit_test_bgd_1_pow_dim_inp_stand_m1_1_50_epochs.npy")

ls_perform = -23.935403167080565

lc_ls_2d = np.array([ls_perform] * len(lc_bgd_2d))

# x_axis = np.arange(len(lc_ls_2d)) // 552
x_axis = np.linspace(0, 50, len(lc_ls_2d))

plt.figure(1)
plt.plot(x_axis, lc_bgd_2d, color='red')
plt.plot(x_axis, lc_ls_2d, color='black', linestyle='--')
plt.xlabel("эпохи", fontsize=fontsize)
plt.ylabel("NMSE, dB", fontsize=fontsize)
# plt.legend(["BGD+Adam", "LS"], fontsize=fontsize)
plt.yticks(np.arange(-24, 3.5, 2.5))
plt.xticks(x_axis[::552 * 5].astype(int))
plt.ylim([-24.5, 0])
plt.grid()

plt.figure(2)
plt.plot(x_axis, lc_bgd_2d, color='red')
plt.plot(x_axis, lc_ls_2d, color='black', linestyle='--')
plt.xlabel("эпохи", fontsize=fontsize)
plt.ylabel("NMSE, dB", fontsize=fontsize)
plt.legend(["BGD+Adam", "LS"], fontsize=fontsize + 4.5)
plt.yticks(np.arange(-24, -20.5, 0.5))
plt.xticks(x_axis[::552 * 5].astype(int))
plt.ylim([-24, -21])
plt.grid()