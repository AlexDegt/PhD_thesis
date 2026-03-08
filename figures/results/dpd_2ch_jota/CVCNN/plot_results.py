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

curr_path = os.getcwd()

seed_0 = 964
epochs = 5000
exp_num = 5
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'simple_cubic_newton']

# In order to plot reproduced results of simulations uncomment reproduced = "reproduced_".
# In order to plot results of simulations, provided by authors uncomment reproduced = "".

reproduced = "reproduced_"
# reproduced = ""

# Determine plot parameters
linewidth = 1
fontsize = 13
xlabel = 'iterations'
ylabel = 'NMSE, dB'
legend = ["LM-MNM", "LM-NM", "CNM", "CMNM"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
# start_points = ["complex", "real", "imag"]
start_points = ["complex"]
yticks_expanded = np.arange(-10, -15, -0.5)
yticks_whole = np.arange(-16, 2, 2)
ylim_expanded = [-15, -10]
ylim_whole = [-15, 1]
yticks = [yticks_whole, yticks_expanded]
ylim = [ylim_whole, ylim_expanded]
figsize=(16, 6)
shade_param = 0.15

# Plot Learning mean, min-max range learning curves for each of the 
# considered training algorithms: LM-MNM, LM-NM, CNM, CMNM, - and 
# for each of the starting points purely real, purely imaginary and complex.

# Function to calculate min-max range curves and mean curves from NMSE curves
def calc_stat_from_nmse(curve, func):
    return 10*np.log10(func(10**(curve/10), axis=0))

lc_all = []
for start_p in start_points:
    lc_train = np.zeros((len(methods), exp_num, epochs + 1))
    lc_aver = np.zeros((len(methods), epochs + 1))
    lc_min = np.zeros((len(methods), epochs + 1))
    lc_max = np.zeros((len(methods), epochs + 1))
    
    for j_method, method in enumerate(methods):
        for exp in range(exp_num):
            exp_name = reproduced + f"paper_exp_{exp}_seed_{seed_0 + exp}_{start_p}_start_{method}_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_{epochs}_epochs"
            add_folder = os.path.join(reproduced + "results")
            curr_path = os.getcwd()
            load_path = os.path.join(curr_path, add_folder, exp_name)
            # Plot learning curve for quality criterion
            content = os.listdir(load_path)
            lc_name = [name for name in content if "lc_qcrit_train_" in name][0]
            lc_train[j_method, exp, :] = np.load(os.path.join(load_path, lc_name))[:epochs + 1]

        lc_aver[j_method, :] = calc_stat_from_nmse(lc_train[j_method, :, :], np.mean)
        lc_min[j_method, :] = calc_stat_from_nmse(lc_train[j_method, :, :], np.min)
        lc_max[j_method, :] = calc_stat_from_nmse(lc_train[j_method, :, :], np.max)

    print(f"Learning curves corresponding to {start_p} starting points:")
    
    # Создание фигуры с GridSpec для контроля положения сабплотов
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)  # wspace регулирует расстояние между графиками
    
    # Создание сабплотов с использованием GridSpec
    ax0 = fig.add_subplot(gs[0])  # Левый график
    ax1 = fig.add_subplot(gs[1])  # Правый график (будет немного правее)
    
    ax = [ax0, ax1]

    for i_graph in range(2):
        for i_method, method in enumerate(methods):
            ax[i_graph].fill_between(
                range(epochs + 1), lc_min[i_method, :], lc_max[i_method, :], 
                color=colors[i_method], alpha=shade_param,
            )
            ax[i_graph].plot(
                lc_aver[i_method, :], color=colors[i_method], linestyle='solid', 
                label=legend[i_method], linewidth=linewidth, 
            )

        ax[i_graph].set_xlabel(xlabel, fontsize=fontsize + 2)
        if i_graph == 0:
            ax[i_graph].set_ylabel(ylabel, fontsize=fontsize + 2)
        ax[i_graph].set_yticks(yticks[i_graph])
        ax[i_graph].set_ylim(ylim[i_graph])
        ax[i_graph].grid()

        if i_graph == 0:
            ax[i_graph].legend(fontsize=fontsize + 4, loc='upper right')

    lc_all.append(lc_train)

    plt.tight_layout()
    plt.show()
    
# Нарисовать вместе RVCNN/CVCCN кривые обучения

# Убрать когда посчитаются эксперименты на 5000 эпох!!
reproduced = ""
epochs = 1500

# Consider learning curves which correspond to CV-CNN
lc_train_cvcnn = np.array(lc_all)[..., :epochs + 1]

methods_rvcnn = ["newton_lev_marq"]
methods_cvcnn = ['simple_cubic_newton']
legend = ["RV-CNN, LM-NM", "CV-CNN, CMNM"]
colors = ["tab:orange", "tab:blue"]

lc_train_rvcnn = np.zeros((len(methods), exp_num, epochs + 1))
lc_aver, lc_min, lc_max = [], [], []

# Function to calculate min-max range curves and mean curves from NMSE curves
def calc_stat_from_nmse(curve, func):
    return 10*np.log10(func(10**(curve/10), axis=0))

# Load and calculate, min, max, average learning curves for RV-CNN
for j_method, method in enumerate(methods_rvcnn):
    for exp in range(exp_num):
        exp_name = reproduced + f"paper_exp_{exp}_seed_{seed_0 + exp}_{methods_rvcnn[0]}_4_channels_6_5_5_2_ker_size_3_3_3_3_act_sigmoid_{epochs}_epochs"
        add_folder = os.path.join(reproduced + "results")
        curr_path = os.path.join(os.getcwd(), "..", "RVCNN")
        load_path = os.path.join(curr_path, add_folder, exp_name)
        # Plot learning curve for quality criterion
        content = os.listdir(load_path)
        lc_name = [name for name in content if "lc_qcrit_train_" in name][0]
        lc_train_rvcnn[j_method, exp, :] = np.load(os.path.join(load_path, lc_name))[:epochs + 1]

    lc_aver.append(calc_stat_from_nmse(lc_train_rvcnn[j_method, :, :], np.mean))
    lc_min.append(calc_stat_from_nmse(lc_train_rvcnn[j_method, :, :], np.min))
    lc_max.append(calc_stat_from_nmse(lc_train_rvcnn[j_method, :, :], np.max))

# Function to calculate min-max range curves and mean curves from NMSE curves
def calc_stat_from_nmse(curve, func):
    return 10*np.log10(func(10**(curve/10), axis=(0, 1)))

# Calculate, min, max, average learning curves for CV-CNN
# Находим индекс метода simple_cubic_newton в methods
method_idx = methods.index(methods_cvcnn[0])
for j_method, method in enumerate(methods_cvcnn):
    lc_aver.append(calc_stat_from_nmse(lc_train_cvcnn[:, method_idx, :, :], np.mean))
    lc_min.append(calc_stat_from_nmse(lc_train_cvcnn[:, method_idx, :, :], np.min))
    lc_max.append(calc_stat_from_nmse(lc_train_cvcnn[:, method_idx, :, :], np.max))
lc_aver = np.array(lc_aver)
lc_min = np.array(lc_min)
lc_max = np.array(lc_max)

# Создание фигуры с GridSpec для контроля положения сабплотов
fig = plt.figure(figsize=figsize)

# Настройка GridSpec с увеличенным расстоянием между графиками
# width_ratios: соотношение ширины колонок (левый график, правый график)
# wspace: расстояние между графиками (чем больше значение, тем больше расстояние)
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)

# Создание сабплотов с использованием GridSpec
ax0 = fig.add_subplot(gs[0])  # Левый график
ax1 = fig.add_subplot(gs[1])  # Правый график
ax = [ax0, ax1]

for i_graph in range(2):
    for i_method in range(len(methods_cvcnn) + len(methods_rvcnn)):
        ax[i_graph].fill_between(
            range(epochs + 1), lc_min[i_method, :], lc_max[i_method, :], 
            color=colors[i_method], alpha=shade_param,
        )
        ax[i_graph].plot(
            lc_aver[i_method, :], color=colors[i_method], linestyle='solid', 
            label=legend[i_method], linewidth=linewidth, 
        )
    ax[i_graph].set_xlabel(xlabel, fontsize=fontsize + 2)
    if i_graph == 0:
        ax[i_graph].set_ylabel(ylabel, fontsize=fontsize + 2)
    ax[i_graph].set_yticks(yticks[i_graph])
    ax[i_graph].set_ylim(ylim[i_graph])
    ax[i_graph].grid()

    if i_graph == 0:
        ax[i_graph].legend(fontsize=fontsize + 4, loc='upper right')

# Дополнительная настройка положения графиков на фигуре
# plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

plt.show()