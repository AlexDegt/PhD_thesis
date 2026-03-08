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
import pickle

plt.rcParams["font.family"] = "Times New Roman"

fontsize = 13

lc_bgd_2d = np.load("lc_qcrit_test_bgd_12_pow_dim_inp_stand_0_1_reg_0_lr_3e_4_epochs_150.npy")

ls_perform = -24.546

lc_ls_2d = np.array([ls_perform] * len(lc_bgd_2d))

# x_axis = np.arange(len(lc_ls_2d)) // 552
x_axis = np.linspace(0, 50, len(lc_ls_2d))

plt.figure(1)
plt.plot(x_axis, lc_bgd_2d, color='red')
plt.plot(x_axis, lc_ls_2d, color='black', linestyle='--')
plt.xlabel("эпохи", fontsize=fontsize)
plt.ylabel("NMSE, dB", fontsize=fontsize)
# plt.legend(["BGD+Adam", "LS"], fontsize=fontsize)
plt.yticks(np.arange(-25, 3.5, 2.5))
plt.xticks(x_axis[::552 * 15].astype(int))
plt.ylim([-25.5, 0])
plt.grid()

plt.figure(2)
plt.plot(x_axis, lc_bgd_2d, color='red')
plt.plot(x_axis, lc_ls_2d, color='black', linestyle='--')
plt.xlabel("эпохи", fontsize=fontsize)
plt.ylabel("NMSE, dB", fontsize=fontsize)
plt.legend(["BGD+Adam", "LS"], fontsize=fontsize + 4.5)
plt.yticks(np.arange(-25, -19, 0.5))
plt.xticks(x_axis[::552 * 15].astype(int))
plt.ylim([-25, -19])
plt.grid()

aclr_perform_ls_path = "aclr_perform_ls_rcond_1e_m40_maxH_alpha_1e_m11.pkl"
aclr_perform_bgd_1730_epochs_path = "aclr_perform_bgd_1730_epochs.pkl"
aclr_perform_bgd_2501_epochs_path = "aclr_perform_bgd_2501_epochs.pkl"
aclr_perform_bgd_14646_epochs_path = "aclr_perform_bgd_14646_epochs.pkl"
aclr_perform_bgd_last_epoch_path = "aclr_perform_bgd_last_epoch.pkl"

with open(aclr_perform_ls_path, "rb") as file:
    data = pickle.load(file)
    aclr_perform_ls = data["ACLR"]
    power_linear = data["power_linear"]
with open(aclr_perform_bgd_1730_epochs_path, "rb") as file:
    aclr_perform_bgd_1730_epochs = pickle.load(file)["ACLR"]
with open(aclr_perform_bgd_2501_epochs_path, "rb") as file:
    aclr_perform_bgd_2501_epochs = pickle.load(file)["ACLR"]
with open(aclr_perform_bgd_14646_epochs_path, "rb") as file:
    aclr_perform_bgd_14646_epochs = pickle.load(file)["ACLR"]
with open(aclr_perform_bgd_last_epoch_path, "rb") as file:
    aclr_perform_bgd_last_epoch = pickle.load(file)["ACLR"]
    
# plt.figure(3)
# plt.plot(power_linear, aclr_perform_bgd_1730_epochs, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_2501_epochs, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_14646_epochs, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_last_epoch, marker='o')
# plt.plot(power_linear, aclr_perform_ls, marker='o', color='black')
# plt.ylim([-56.5, -38])
# plt.yticks(np.arange(-56, -37, 1))
# plt.xticks(np.arange(0, 1, 0.1))
# plt.ylabel("ACLR, дБ", fontsize=fontsize)
# # plt.xlabel("Выходная мощность УМ, Вт", fontsize=fontsize)
# plt.legend(["BGD+Adam, 3 эпохи",
#             "BGD+Adam, 5 эпох",
#             "BGD+Adam, 27 эпох",
#             "BGD+Adam, 150 эпох",
#             "LS",], fontsize=fontsize)
# plt.grid()

# plt.figure(4)
# plt.plot(power_linear, aclr_perform_bgd_1730_epochs - aclr_perform_ls, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_2501_epochs - aclr_perform_ls, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_14646_epochs - aclr_perform_ls, marker='o')
# plt.plot(power_linear, aclr_perform_bgd_last_epoch - aclr_perform_ls, marker='o')
# plt.ylim([-0.5, 11])
# plt.yticks(np.arange(0, 11, 1))
# plt.xticks(np.arange(0, 1, 0.1))
# plt.ylabel(r"$\Delta$ ACLR, дБ", fontsize=fontsize)
# plt.xlabel("Выходная мощность УМ, Вт", fontsize=fontsize)
# plt.legend([r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 3\ эпохи}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
#             r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 5\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
#             r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 27\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
#             r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 150\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
#             ], fontsize=fontsize)
# plt.grid()

# Создаем фигуру с двумя подграфиками друг под другом
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 строки, 1 колонка

# Первый график (верхний)
ax1.plot(power_linear, aclr_perform_bgd_1730_epochs, marker='o')
ax1.plot(power_linear, aclr_perform_bgd_2501_epochs, marker='o')
ax1.plot(power_linear, aclr_perform_bgd_14646_epochs, marker='o')
ax1.plot(power_linear, aclr_perform_bgd_last_epoch, marker='o')
ax1.plot(power_linear, aclr_perform_ls, marker='o', color='black')
ax1.set_ylim([-56.5, -36])
ax1.set_yticks(np.arange(-56, -35, 2))
ax1.set_xticks(np.arange(0, 1, 0.1))
ax1.set_ylabel("ACLR, дБ", fontsize=fontsize)
ax1.legend([r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 3\ эпохи}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 5\ эпох}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 27\ эпох}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 150\ эпох}}$",
            r"$\mathrm{ACLR}_{\mathrm{LS}}$",
            ], fontsize=fontsize, loc="upper right", bbox_to_anchor=(0.93, 1))
ax1.grid(True)

# Второй график (нижний)
ax2.plot(power_linear, aclr_perform_bgd_1730_epochs - aclr_perform_ls, marker='o')
ax2.plot(power_linear, aclr_perform_bgd_2501_epochs - aclr_perform_ls, marker='o')
ax2.plot(power_linear, aclr_perform_bgd_14646_epochs - aclr_perform_ls, marker='o')
ax2.plot(power_linear, aclr_perform_bgd_last_epoch - aclr_perform_ls, marker='o')
ax2.set_ylim([-0.5, 15])
ax2.set_yticks(np.arange(0, 15, 1))
ax2.set_xticks(np.arange(0, 1, 0.1))
ax2.set_ylabel(r"$\Delta$ ACLR, дБ", fontsize=fontsize)
ax2.set_xlabel("Выходная мощность УМ, Вт", fontsize=fontsize)
ax2.legend([r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 3\ эпохи}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 5\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 27\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
            r"$\mathrm{ACLR}_{\mathrm{BGD+Adam, 150\ эпох}}$ - $\mathrm{ACLR}_{\mathrm{LS}}$",
            ], fontsize=fontsize, loc="upper right", bbox_to_anchor=(0.93, 1))
ax2.grid(True)

# Настройка отступов, чтобы подписи не налезали друг на друга
plt.tight_layout()

# Показать график
plt.show()