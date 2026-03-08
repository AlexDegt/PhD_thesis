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

aclr_perform_ls_path = "aclr_perform_ls_rcond_1e_m40_maxH_alpha_1e_m11.pkl"
aclr_perform_rank_1_path = "aclr_perform_rank_1.pkl"
aclr_perform_rank_2_path = "aclr_perform_rank_2.pkl"
aclr_perform_rank_3_path = "aclr_perform_rank_3.pkl"

with open(aclr_perform_ls_path, "rb") as file:
    data = pickle.load(file)
    aclr_perform_ls = data["ACLR"]
    power_linear = data["power_linear"]
with open(aclr_perform_rank_1_path, "rb") as file:
    aclr_perform_rank_1 = pickle.load(file)["ACLR"]
with open(aclr_perform_rank_2_path, "rb") as file:
    aclr_perform_rank_2 = pickle.load(file)["ACLR"]
with open(aclr_perform_rank_3_path, "rb") as file:
    aclr_perform_rank_3 = pickle.load(file)["ACLR"]

# Создаем фигуру с двумя подграфиками друг под другом
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 строки, 1 колонка

# Первый график (верхний)
ax1.plot(power_linear, aclr_perform_rank_1, marker='o')
ax1.plot(power_linear, aclr_perform_rank_2, marker='o')
ax1.plot(power_linear, aclr_perform_rank_3, marker='o')
ax1.plot(power_linear, aclr_perform_ls, marker='o', color='black')
ax1.set_ylim([-56.5, -45])
ax1.set_yticks(np.arange(-56, -45, 1))
ax1.set_xticks(np.arange(0, 1, 0.1))
ax1.set_ylabel("ACLR, дБ", fontsize=fontsize)
ax1.legend([r"$\mathrm{ACLR}_{\mathrm{Ранг\ 1,\ MNM}}$",
            r"$\mathrm{ACLR}_{\mathrm{Ранг\ 2,\ MNM}}$",
            r"$\mathrm{ACLR}_{\mathrm{Ранг\ 3,\ MNM}}$",
            r"$\mathrm{ACLR}_{\mathrm{2D\ модель,\ LS}}$",
            ], fontsize=fontsize, loc="upper right", bbox_to_anchor=(0.93, 1))
ax1.grid(True)

# Второй график (нижний)
ax2.plot(power_linear, aclr_perform_rank_1 - aclr_perform_ls, marker='o')
ax2.plot(power_linear, aclr_perform_rank_2 - aclr_perform_ls, marker='o')
ax2.plot(power_linear, aclr_perform_rank_3 - aclr_perform_ls, marker='o')
ax2.set_ylim([-0.5, 10])
ax2.set_yticks(np.arange(0, 10, 1))
ax2.set_xticks(np.arange(0, 1, 0.1))
ax2.set_ylabel(r"$\Delta$ ACLR, дБ", fontsize=fontsize)
ax2.set_xlabel("Выходная мощность УМ, Вт", fontsize=fontsize)
ax2.legend([r"$\mathrm{ACLR}_{\mathrm{Ранг\ 1,\ MNM}}$ - $\mathrm{ACLR}_{\mathrm{2D\ модель,\ LS}}$",
            r"$\mathrm{ACLR}_{\mathrm{Ранг\ 2,\ MNM}}$ - $\mathrm{ACLR}_{\mathrm{2D\ модель,\ LS}}$",
            r"$\mathrm{ACLR}_{\mathrm{Ранг\ 3,\ MNM}}$ - $\mathrm{ACLR}_{\mathrm{2D\ модель,\ LS}}$",
            ], fontsize=fontsize, loc="upper right", bbox_to_anchor=(0.93, 1))
ax2.grid(True)

# Настройка отступов, чтобы подписи не налезали друг на друга
plt.tight_layout()

# Показать график
plt.show()