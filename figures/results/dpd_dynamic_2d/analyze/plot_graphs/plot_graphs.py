import sys, os
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

def aclr_fn(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
    """ 
        Calculate Adjacent Channel Leakage Ratio
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    ind1 = (nfft // 2) - int(np.ceil(nfft * f/fs))
    ind2 = (nfft // 2) + int(np.ceil(nfft * f/fs))
    guard = 4 # 10
    aclr = np.sum(psd[ind2 + guard: 2 * ind2 - ind1 + guard])/np.sum(psd[ind1: ind2])
    return 10 * np.log10(aclr)

def get_psd(sig, Fs=1.0, nfft=2048, window='blackman', nperseg=None, noverlap=None, scale='log'):
    """ 
        Returns Power Spectral Density of input signal sig
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, Fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    if scale == 'log':
        return freqs, 10*np.log10(np.fft.fftshift(psd))
    elif scale == 'lin':
        return freqs, np.fft.fftshift(psd)
    else:
        raise ValueError(f"scale parameter must equal \'log\' or \'lin\', but {scale} is given.")

def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=None, is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, 
             noverlap=None, y_shift=0, color=None, fontsize=13, xlabel='frequency', ylabel='Magnitude [dB]', scale='log'):
    """ Plotting power spectral density """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()
        
    if isinstance(color, list):
        assert len(signals) == len(color) or (len(color) == 1)
        for c in color:
            assert isinstance(c, str)
        if len(color) == 1:
            color *= len(signals)
    elif isinstance(color, str) or color is None:
        color = [color] * len(signals)
    else:
        raise TypeError(f"color parameter must be either of a str or list of str type, but {type(color)} is given")
      
    ax.set_xlabel(xlabel, fontsize=fontsize)
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)

    for j_sig, iisignal in enumerate(signals):
        # freqs = np.linspace(-Fs/2, Fs/2, iisignal.size)
        # plt.plot(freqs, 10*np.log10(np.fft.fftshift(np.fft.fft(iisignal))))

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, 1, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)*Fs
        freqs += xshift
        if scale == 'log':
            psd = 10*np.log10(np.fft.fftshift(psd)) + y_shift
        elif scale == 'lin':
            psd =  np.fft.fftshift(psd) + y_shift
        else:
            raise ValueError(f"scale parameter must equal \'log\' or \'lin\', but {scale} is given.")
        ax_ptr, = ax.plot(freqs, psd, color=color[j_sig])
#        ax_ptr, = ax.plot(freqs, psd, color='tab:blue')

    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if legend is not None:
        ax.legend(legend, fontsize=fontsize)
    if is_save:
        nfig.savefig(filename)
    plt.show()
    return ax_ptr

# Determine powers for model input for train and test correspondingly
out_power_raugh = np.array([-11.6, -10.5, -9.5, -8.5, -7.6, -6.8, -6, -5.2, -4.5, -3.8, -3.1, -2.5, -1.9, -1.3, -0.8, -0.4])
in_power_ref = 10 ** (np.arange(-16, 0, 1) / 10)
out_power_ref = 10 ** ((30 + out_power_raugh) / 10)
in_power = list(10 ** ((np.load('pa_powers_round.npy') - 1)/ 10))
# out_powers in W
pa_powers = np.interp(in_power, in_power_ref, out_power_ref) / 1000

power_cases_train = pa_powers[::2]
power_cases_test = pa_powers[1::2]

# Determine parameters of the simulation, which is chosen for PSD and ACLR(pa_power) graphs:
pow_param_num = 10
param_num = 22
delay_num = 4
slot_num = 4

f = 10 / 1000 # GHz
fs = 245.76 / 1000 # GHz
nfft = 512

figure = 1
ylim = [-50, -10]

# one_dim_folder = os.path.join("..", f"{1}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "36_param_4_slot_61_cases_27_delay")
# two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", f"{param_num}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
one_dim_folder = os.path.join("..", f"ls_1_pow_dim_inp_stand_m1_1_reg_m13")
two_dim_folder = os.path.join("..", f"ls_12_pow_dim_inp_stand_m1_1_reg_m13")
# two_dim_folder = one_dim_folder

for j_folder, folder in enumerate([one_dim_folder, two_dim_folder]):

    tx_train = np.load(os.path.join(folder, 'x.npy'))
    pa_out_train = np.load(os.path.join(folder, 'd.npy'))
    model_out_train = np.load(os.path.join(folder, 'y.npy'))
    tx_test = np.load(os.path.join(folder, 'x_test.npy'))
    pa_out_test = np.load(os.path.join(folder, 'd_test.npy'))
    model_out_test = np.load(os.path.join(folder, 'y_test.npy'))
    
    sig_len_train = int(len(tx_train) / len(power_cases_train))
    sig_len_test = int(len(tx_test) / len(power_cases_test))
    
    aclr_train = []
    aclr_test = []
    
    psd_train, psd_test, psd = [], [], []
    
    for i in range(len(power_cases_train)):
        x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
        d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
        y = model_out_train[i * sig_len_train: (i + 1) * sig_len_train]
        scale = np.max(abs(x))
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_train.append(aclr_val)
        freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
        psd_train.append(psd)
        print(f"Case train {power_cases_train[i]:.2f} W: ACLR = {aclr_val} dB")

        alpha = 1
        fontsize = 18
        if j_folder == 0:
            plt.figure(450)          
            plot_psd(x + d, nfig=450, clf=False, nfft=int(nfft/alpha), color='red', Fs=fs, xlabel='частота, ГГц', ylabel='СПМ, дБм/ГГц', fontsize=fontsize, legend=['TX, not corrected'])
            plot_psd(x + d - y, nfig=450, clf=False, nfft=int(nfft/alpha), color='blue', Fs=fs, xlabel='частота, ГГц', ylabel='СПМ, дБм/ГГц', fontsize=fontsize, legend=['TX, corrected'])
        
            freqs, psd_nc = get_psd(x + d, Fs=fs, nfft=int(nfft/alpha), scale='lin')            
            freqs, psd_1d = get_psd(x + d - y, Fs=fs, nfft=int(nfft/alpha), scale='lin')
            
            freqs = freqs[::int(alpha)]
            psd_nc = psd_nc[::int(alpha)]
            psd_1d = psd_1d[::int(alpha)]
            
            # plt.plot(freqs, 10 * np.log10(psd_nc), color='red')
            # plt.plot(freqs, 10 * np.log10(psd_1d), color='blue')
            plt.grid(which='major', linestyle='--', linewidth=0.85, alpha=1, color='black')
            plt.gca().xaxis.grid(True, linestyle=(0, (4, 4)))
            plt.gca().yaxis.grid(True, linestyle=(0, (4, 4))) 
            plt.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0)
            plt.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            # Включаем дополнительные тики 
            plt.minorticks_on()
            
            psd_nc = np.concatenate([freqs[None, :], 10 * np.log10(psd_nc)[None, :]], axis=0).T
            psd_1d = np.concatenate([freqs[None, :], 10 * np.log10(psd_1d)[None, :]], axis=0).T
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', 'nc', f'{2*i}.txt'), psd_nc)
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', '1d', f'{2*i}.txt'), psd_1d)
        
        if j_folder == 1:
            plt.figure(450)
            plot_psd(x + d - y, xshift=1.8, nfig=450, clf=False, nfft=int(nfft/alpha), color='forestgreen', Fs=fs, xlabel='freq, MHz', ylabel='PSD, dBm/MHz', fontsize=fontsize, legend=['TX, DPD off', 'TX, DPD on, 1-dim. model', 'TX, DPD, on, 2-dim. model'])
        
            freqs, psd_2d = get_psd(x + d - y, Fs=fs, nfft=int(nfft/alpha), scale='lin')
            
            freqs = freqs[::int(alpha)] + 1.8
            psd_2d = psd_2d[::int(alpha)]
            
            # plt.plot(freqs, 10 * np.log10(psd_2d), color='green')
            
            psd_2d = np.concatenate([freqs[None, :], 10 * np.log10(psd_2d)[None, :]], axis=0).T
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', '2d', f'{2*i}.txt'), psd_2d)
    
    for i in range(len(power_cases_test)):
        x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
        d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        y = model_out_test[i * sig_len_test: (i + 1) * sig_len_test]
        scale = np.max(abs(x))
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_test.append(aclr_val)
        freqs, psd = get_psd(x + d - y, Fs=fs, nfft=nfft)
        psd_test.append(psd)
        print(f"Case test {power_cases_test[i]:.2f} W: ACLR = {aclr_val} dB")

        fontsize = 18
        if j_folder == 0:
            plt.figure(450)          
            plot_psd(x + d, nfig=450, xshift=1.8, clf=False, nfft=int(nfft/alpha), color='red', Fs=fs, xlabel='freq, MHz', ylabel='PSD, dBm/MHz', fontsize=fontsize, legend=['TX, not corrected'])
            plot_psd(x + d - y, nfig=450, xshift=1.8, clf=False, nfft=int(nfft/alpha), color='blue', Fs=fs, xlabel='freq, MHz', ylabel='PSD, dBm/MHz', fontsize=fontsize, legend=['TX, corrected'])
        
            freqs, psd_nc = get_psd(x + d, Fs=fs, nfft=int(nfft/alpha), scale='lin')            
            freqs, psd_1d = get_psd(x + d - y, Fs=fs, nfft=int(nfft/alpha), scale='lin')
            
            freqs = freqs[::int(alpha)]
            psd_nc = psd_nc[::int(alpha)]
            psd_1d = psd_1d[::int(alpha)]
            
            # plt.plot(freqs, 10 * np.log10(psd_nc), color='red')
            # plt.plot(freqs, 10 * np.log10(psd_1d), color='blue')
            
            psd_nc = np.concatenate([freqs[None, :], 10 * np.log10(psd_nc)[None, :]], axis=0).T
            psd_1d = np.concatenate([freqs[None, :], 10 * np.log10(psd_1d)[None, :]], axis=0).T
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', 'nc', f'{2*i}.txt'), psd_nc)
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', '1d', f'{2*i}.txt'), psd_1d)
        
        if j_folder == 1:
            plt.figure(450)
            plot_psd(x + d - y, nfig=450, xshift=1.8, clf=False, nfft=int(nfft/alpha), color='forestgreen', Fs=fs, xlabel='freq, MHz', ylabel='PSD, dBm/MHz', fontsize=fontsize, legend=['TX, DPD выкл.', 'TX, DPD вкл., 1D модель', 'TX, DPD вкл., 2D модель'])
        
            freqs, psd_2d = get_psd(x + d - y, Fs=fs, nfft=int(nfft/alpha), scale='lin')
            
            # plt.plot(freqs, 10 * np.log10(psd_2d), color='green')
            
            freqs = freqs[::int(alpha)]
            psd_2d = psd_2d[::int(alpha)]
            
            psd_2d = np.concatenate([freqs[None, :], 10 * np.log10(psd_2d)[None, :]], axis=0).T
            np.savetxt(os.path.join('psd', 'nc_1d_2d_all_powers', '2d', f'{2*i}.txt'), psd_2d)
       
        legend_lines = [
                Line2D([0], [0], color='red', lw=2, label='DPD выкл.'),
                Line2D([0], [0], color='blue', lw=2, label='DPD вкл., 1D'),
                Line2D([0], [0], color='forestgreen', lw=2, label='DPD вкл., 2D'),
            ]
        legend = plt.legend(handles=legend_lines, fontsize=14.5, loc='lower center', labelspacing=0.34, 
                    framealpha=1, edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(0.5)
        
        plt.xticks(np.arange(1.8 -2 * 0.140, 1.8 + 0.140, 0.040))
        plt.tick_params(axis='both', labelsize=15)
        plt.tick_params(axis='both', which='minor', length=5) 
        plt.tick_params(axis='both', which='major', length=7.5) 
        plt.xlim([1.8 -0.12288, 1.8+0.12288])
        plt.ylim([-85, 5])
        plt.xlabel('частота, ГГц', fontsize=18, fontweight='light' )
        plt.ylabel('СПМ, дБм/ГГц', fontsize=18, fontweight='light')
    
    psd = list(np.zeros((len(pa_powers),)))
    psd[::2] = psd_train
    psd[1::2] = psd_test
    psd = np.array(psd)
    
    if j_folder == 0:
        psd_train_nc, psd_test_nc, psd_nc = [], [], []
        aclr_train_no_correct, aclr_test_no_correct = [], []
        # Calculate ACLR for PA output without correction
        for i in range(len(power_cases_train)):
            x = tx_train[i * sig_len_train: (i + 1) * sig_len_train]
            d = pa_out_train[i * sig_len_train: (i + 1) * sig_len_train]
            scale = np.max(abs(x))
            x /= scale
            d /= scale
            aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
            aclr_train_no_correct.append(aclr_val)
            freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
            psd_train_nc.append(psd_val)
        
        for i in range(len(power_cases_test)):
            x = tx_test[i * sig_len_test: (i + 1) * sig_len_test]
            d = pa_out_test[i * sig_len_test: (i + 1) * sig_len_test]
            scale = np.max(abs(x))
            x /= scale
            d /= scale
            aclr_val = aclr_fn(x + d, f=f, fs=fs, nfft=nfft)
            aclr_test_no_correct.append(aclr_val)
            freqs, psd_val = get_psd(x + d, Fs=fs, nfft=nfft)
            psd_test_nc.append(psd_val)
        
        psd_nc = list(np.zeros((len(pa_powers),)))
        psd_nc[::2] = psd_train_nc
        psd_nc[1::2] = psd_test_nc
        psd_nc = np.array(psd_nc)
        
        aclr_no_correct = list(np.zeros((len(pa_powers),)))
        aclr_no_correct[::2] = aclr_train_no_correct
        aclr_no_correct[1::2] = aclr_test_no_correct
        
        plt.figure(figure)
        plt.plot(pa_powers, aclr_no_correct, marker='o', color='black')
        
        # Concatenate arrays to write in .txt file
        aclr_no_correct_tmp = []
        aclr_no_correct_tmp.append(pa_powers)
        aclr_no_correct_tmp.append(aclr_no_correct)
        aclr_no_correct_tmp = np.array(aclr_no_correct_tmp).T
        # Save arrays to .txt file to draw picture in latex
        try:
            folder_name = "performance"
            os.mkdir(folder_name)
        except:
            pass
        np.savetxt(os.path.join(folder_name, 'aclr_no_correct.txt'), aclr_no_correct_tmp)
    
        one_dim_sep_folder = os.path.join("..", f"{1}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately")
        aclr_one_dim_sep = []
        for k in range(len(pa_powers)):
            curr_one_dim_sep_folder = os.path.join(one_dim_sep_folder, f"36_param_4_slot_61_cases_27_delay_power_{k}")
            x = np.load(os.path.join(curr_one_dim_sep_folder, "x.npy"))
            d = np.load(os.path.join(curr_one_dim_sep_folder, "d.npy"))
            y = np.load(os.path.join(curr_one_dim_sep_folder, "y.npy"))
            aclr_one_dim_sep.append(aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft))
            # plt.figure(600)
            # plot_psd(x + d, x + d - y, clf=False)
    
    plt.figure(figure)
    # if j_folder == 0:
    #     plt.plot(pa_powers, aclr_one_dim_sep, marker='o')
    plt.plot(pa_powers[::2], aclr_train, marker='o')
    plt.plot(pa_powers[1::2], aclr_test, marker='o')
    plt.xlabel('PA output power, W', fontsize=13)
    plt.ylabel('ACLR, dB', fontsize=13)
    plt.ylim(ylim)
    plt.yticks(np.arange(15, -60, -5))
    
    # Concatenate arrays to write in .txt file
    aclr_correct_train_tmp, aclr_correct_test_tmp = [], []
    aclr_correct_train_tmp.append(pa_powers[::2])
    aclr_correct_test_tmp.append(pa_powers[1::2])
    aclr_correct_train_tmp.append(aclr_train)
    aclr_correct_test_tmp.append(aclr_test)
    aclr_correct_train_tmp = np.array(aclr_correct_train_tmp).T
    aclr_correct_test_tmp = np.array(aclr_correct_test_tmp).T
    # Save arrays to .txt file to draw picture in latex
    np.savetxt(os.path.join(folder_name, f'aclr_correct_{j_folder + 1}_dim_train.txt'), aclr_correct_train_tmp)
    np.savetxt(os.path.join(folder_name, f'aclr_correct_{j_folder + 1}_dim_test.txt'), aclr_correct_test_tmp)
    
plt.legend(['DPD off',
            # 'DPD on, apply 1D model separately',
            'DPD on, 1D model, train', 
            'DPD on, 1D model, test',
            'DPD on, 2D model, train',
            'DPD on, 2D model, test'], fontsize=13, loc='upper right')
plt.grid()

coef = -8#21
psd += coef
psd_nc += coef

psd_min = -52 + coef
psd_max = 10 + coef
psd[psd < psd_min] = psd_min
# psd[psd >= -46] = -46
psd_nc[psd_nc < psd_min] = psd_min
# psd_nc[psd_nc >= -46] = -46

fontsize = 17

# Draw PSD color map
pa_range = np.arange(psd_min - 1, psd_max, 1)
levels = list(pa_range)
plt.figure(figure * 100)
F, P = np.meshgrid(freqs + 1.8, pa_powers)
cs = plt.contourf(F, P, psd, levels=levels, cmap ="jet")
cbar = plt.colorbar(cs) 
cbar.set_label("СПМ, дБм/ГГц", rotation=270, labelpad=15, fontsize=fontsize)
plt.ylabel("Выходная мощность УМ, Вт", fontsize=fontsize)
plt.xlabel("частота, MГц", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=13.8)
cbar.ax.tick_params(labelsize=13.8)
plt.xticks(np.arange(-2 * 0.140 + 1.8, 0.140 + 1.8, 0.040))
plt.xlim([-0.12288 + 1.8, 0.12288 + 1.8])

levels = list(pa_range)
plt.figure(figure * 200)
F, P = np.meshgrid(freqs + 1.8, pa_powers)
cs = plt.contourf(F, P, psd_nc, levels=levels, cmap ="jet")
cbar = plt.colorbar(cs)
cbar.set_label("СПМ, дБм/ГГц", rotation=270, labelpad=15, fontsize=fontsize)
plt.ylabel("Выходная мощность УМ, Вт", fontsize=fontsize)
plt.xlabel("частота, MГц", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=13.8)
cbar.ax.tick_params(labelsize=13.8)
plt.xticks(np.arange(-2 * 0.140 + 1.8, 0.140 + 1.8, 0.040))
plt.xlim([-0.12288 + 1.8, 0.12288 + 1.8])

# Save arrays to .txt file to draw picture in latex
try:
    folder_name = "psd"
    os.mkdir(folder_name)
except:
    pass
np.savetxt(os.path.join(folder_name, 'freq.txt'), F)
np.savetxt(os.path.join(folder_name, 'powers.txt'), P)
np.savetxt(os.path.join(folder_name, 'PSD_corrected.txt'), psd)
np.savetxt(os.path.join(folder_name, 'PSD_not_corrected.txt'), psd_nc)

# Calculate ACLR for 2D model w.r.t. the number of parameters per |x| dimension:
param_num = list(np.arange(2, 32, 2))
aclr_val_param_depend_train, aclr_val_param_depend_test = [], []

two_dim_folder = os.path.join("..", f"{pow_param_num}_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm")

for p in param_num:
    folder = os.path.join(two_dim_folder, f"{p}_param_{slot_num}_slot_61_cases_{delay_num}_delay")
    
    tx_train = np.load(os.path.join(folder, 'x.npy'))
    pa_out_train = np.load(os.path.join(folder, 'd.npy'))
    model_out_train = np.load(os.path.join(folder, 'y.npy'))
    
    tx_test = np.load(os.path.join(folder, 'x_test.npy'))
    pa_out_test = np.load(os.path.join(folder, 'd_test.npy'))
    model_out_test = np.load(os.path.join(folder, 'y_test.npy'))
    
    sig_len_train = int(len(tx_train) / len(power_cases_train))
    sig_len_test = int(len(tx_test) / len(power_cases_test))
    
    aclr_val_curr_param_depend = []
    for j_case in range(len(power_cases_train)):
        x = tx_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
        d = pa_out_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
        y = model_out_train[j_case * sig_len_train: (j_case + 1) * sig_len_train]
    
        scale = np.max(abs(x))
        
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_val_curr_param_depend.append(aclr_val)
    aclr_val_param_depend_train.append(aclr_val_curr_param_depend)
    
    aclr_val_curr_param_depend = []
    for j_case in range(len(power_cases_test)):
        x = tx_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
        d = pa_out_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
        y = model_out_test[j_case * sig_len_test: (j_case + 1) * sig_len_test]
    
        scale = np.max(abs(x))
        
        x /= scale
        d /= scale
        y /= scale
        aclr_val = aclr_fn(x + d - y, f=f, fs=fs, nfft=nfft)
        aclr_val_curr_param_depend.append(aclr_val)
    aclr_val_param_depend_test.append(aclr_val_curr_param_depend)
    
aclr_val_param_depend_train = np.array(aclr_val_param_depend_train).T
aclr_val_param_depend_test = np.array(aclr_val_param_depend_test).T

aclr_val_param_depend = np.zeros((len(pa_powers), len(param_num)))
# aclr_val_param_depend = np.concatenate([aclr_val_param_depend_train, aclr_val_param_depend_test], axis=0)
aclr_val_param_depend[::2] = aclr_val_param_depend_train
aclr_val_param_depend[1::2] = aclr_val_param_depend_test

plt.figure(figure * 300)
for j_param, p in enumerate(pa_powers):
    plt.plot(param_num, aclr_val_param_depend[j_param, :], marker='o')
# plt.plot(param_num, aclr_val_param_depend[-1, :], color='blue', marker='o')
# plt.plot(param_num, aclr_val_param_depend[1, :], color='red', marker='o')
plt.yticks(np.arange(-30, -54, -2))
plt.xticks(param_num)
plt.xlabel(r"Number of parameters per magnitude dimension $P_{1}$", fontsize=13)
plt.ylabel("ACLR, dB", fontsize=13)
# plt.legend(['PA output power: 0.912 W',
#             'PA output power: 0.069 W'], fontsize=13)
plt.grid()
# Concatenate axis to save to .txt file
param_num = np.array(param_num)
aclr_val_hp_param_depend = np.array(aclr_val_param_depend[-1, :])
aclr_val_lp_param_depend = np.array(aclr_val_param_depend[0, :])
param_curve_high_power = np.concatenate([param_num[None, :], aclr_val_hp_param_depend[None, :]], axis=0).T
param_curve_low_power = np.concatenate([param_num[None, :], aclr_val_lp_param_depend[None, :]], axis=0).T
# Save arrays to .txt file to draw picture in latex
try:
    folder_name = "param_curve"
    os.mkdir(folder_name)
except:
    pass
np.savetxt(os.path.join(folder_name, 'param_curve_high_power.txt'), param_curve_high_power)
np.savetxt(os.path.join(folder_name, 'param_curve_low_power.txt'), param_curve_low_power)

'''
    Plot PSD curves of application of parameters corresponding one case to another data case
'''

folder_case_m2_3dBm = os.path.join("..", "1_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately", "36_param_4_slot_61_cases_27_delay_power_25")
folder_case_m11_6dBm = os.path.join("..", "1_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm", "each_case_separately", "36_param_4_slot_61_cases_27_delay_power_0")

x_m2_3 = np.load(os.path.join(folder_case_m2_3dBm, "x.npy"))
d_m2_3 = np.load(os.path.join(folder_case_m2_3dBm, "d.npy"))
y_m2_3_to_m2_3 = np.load(os.path.join(folder_case_m2_3dBm, "y.npy"))
y_m1_3_to_m2_3 = np.load(os.path.join(folder_case_m2_3dBm, "y_param_-2.9dBm_apply_to_-4.6dBm.npy"))

x_m11_6 = np.load(os.path.join(folder_case_m11_6dBm, "x.npy"))
d_m11_6 = np.load(os.path.join(folder_case_m11_6dBm, "d.npy"))
y_m11_6_to_m11_6 = np.load(os.path.join(folder_case_m11_6dBm, "y.npy"))
y_m0_4_to_m11_6 = np.load(os.path.join(folder_case_m11_6dBm, "y_param_-1.0dBm_apply_to_-16.0dBm.npy"))

# Calculate coefficients for all signals to satisfy signal power
freqs, psd1 = get_psd(x_m2_3 + d_m2_3, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_2_3_dpd_off = 10 ** (-2.3 / 10) / interg
print(f"Signal power 0.589 W, DPD off: {interg}")
# plt.figure(12)
# plt.plot(freqs, psd1 * alpha_2_3_dpd_off)
# plt.figure(13)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_2_3_dpd_off))

freqs, psd1 = get_psd(x_m2_3 + d_m2_3 - y_m2_3_to_m2_3, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_2_3_dpd_on_appl_2_3 = 10 ** (-2.3 / 10) / interg
print(f"Signal power 0.589 W, DPD on, apply 0.589 W param: {interg}")
# plt.figure(12)
# plt.plot(freqs, psd1 * alpha_2_3_dpd_on_appl_2_3)
# plt.figure(13)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_2_3_dpd_on_appl_2_3))

freqs, psd1 = get_psd(x_m2_3 + d_m2_3 - y_m1_3_to_m2_3, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_2_3_dpd_on_appl_1_3 = 10 ** (-2.3 / 10) / interg
print(f"Signal power 0.589 W, DPD on, apply 0.741 W param: {interg}")
# plt.figure(12)
# plt.plot(freqs, psd1 * alpha_2_3_dpd_on_appl_1_3)
# plt.figure(13)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_2_3_dpd_on_appl_1_3))

# Calculate coefficients for all signals to satisfy signal power
freqs, psd1 = get_psd(x_m11_6 + d_m11_6, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_11_6_dpd_off = 10 ** (-11.6 / 10) / interg
print(f"Signal power 0.069 W, DPD off: {interg}")
# plt.figure(22)
# plt.plot(freqs, psd1 * alpha_11_6_dpd_off)
# plt.figure(23)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_11_6_dpd_off))

freqs, psd1 = get_psd(x_m11_6 + d_m11_6 - y_m11_6_to_m11_6, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_11_6_dpd_on_appl_11_6 = 10 ** (-11.6 / 10) / interg
print(f"Signal power 0.069 W, DPD on, apply 0.069 W param: {interg}")
# plt.figure(22)
# plt.plot(freqs, psd1 * alpha_11_6_dpd_on_appl_11_6)
# plt.figure(23)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_11_6_dpd_on_appl_11_6))

freqs, psd1 = get_psd(x_m11_6 + d_m11_6 - y_m0_4_to_m11_6, Fs=fs, nfft=nfft, scale='lin')
interg = simps(psd1, freqs)
alpha_11_6_dpd_on_appl_0_4 = 10 ** (-11.6 / 10) / interg
print(f"Signal power 0.069 W, DPD on, apply 0.912 W param: {interg}")
# plt.figure(22)
# plt.plot(freqs, psd1 * alpha_11_6_dpd_on_appl_0_4)
# plt.figure(23)
# plt.plot(freqs, 10 * np.log10(psd1 * alpha_11_6_dpd_on_appl_0_4))

legend=['DPD выкл.', 'DPD вкл. Обучить на 0.589 Вт. Применить к 0.589 Вт', 'DPD вкл. Обучить на 0.741 Вт. Применить к 0.589 Вт']
color = ['red', 'green', 'blue']
fontsize = 14
freqs, psd1 = get_psd(x_m2_3 + d_m2_3, Fs=fs, nfft=nfft, scale='lin')
freqs, psd2 = get_psd(x_m2_3 + d_m2_3 - y_m2_3_to_m2_3, Fs=fs, nfft=nfft, scale='lin')
freqs, psd3 = get_psd(x_m2_3 + d_m2_3 - y_m1_3_to_m2_3, Fs=fs, nfft=nfft, scale='lin')
plt.figure(30)
plt.plot(freqs + 1.8, 10 * np.log10(psd1 * alpha_2_3_dpd_off), color=color[0])
plt.plot(freqs + 1.8, 10 * np.log10(psd2 * alpha_2_3_dpd_on_appl_2_3), color=color[1])
plt.plot(freqs + 1.8, 10 * np.log10(psd3 * alpha_2_3_dpd_on_appl_1_3), color=color[2])
plt.xlabel('частота, ГГц', fontsize=fontsize)
plt.ylabel('СПМ, дБм/ГГц', fontsize=fontsize)
plt.legend(legend, fontsize=16, loc='lower center')
plt.yticks(np.arange(-90, 30, 10))
plt.ylim([-85, 15])
plt.xticks(np.arange(-2 * 0.120 + 1.8, 0.140 + 1.8, 0.040))
plt.xlim([-0.12288 + 1.8, 0.12288 + 1.8])
plt.grid()
aclr_train = aclr_fn(x_m2_3 + d_m2_3 - y_m2_3_to_m2_3, f=f, fs=fs)
aclr_test = aclr_fn(x_m2_3 + d_m2_3 - y_m1_3_to_m2_3, f=f, fs=fs)
print(f"TX DPD on. Train on -2.3 dBm. Apply to -2.3 dBm, ACLR = {aclr_train} dB")
print(f"TX DPD on. Train on -1.3 dBm. Apply to -2.3 dBm, ACLR = {aclr_test} dB")
psd1_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd1 * alpha_2_3_dpd_off)[None, :] - 30.5], axis=0).T
psd2_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd2 * alpha_2_3_dpd_on_appl_2_3)[None, :] - 30.5], axis=0).T
psd3_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd3 * alpha_2_3_dpd_on_appl_1_3)[None, :] - 30.5], axis=0).T
np.savetxt(os.path.join('psd', 'psd_dpd_off_0_589W.txt'), psd1_tmp)
np.savetxt(os.path.join('psd', 'psd_dpd_on_0_589Wto0_589W.txt'), psd2_tmp)
np.savetxt(os.path.join('psd', 'psd_dpd_on_0_741Wto0_589W.txt'), psd3_tmp)

legend=['DPD выкл.', 'DPD вкл. Обучить на 0.069 Вт. Применить к 0.069 Вт', 'DPD вкл. Обучить на 0.912 Вт. Применить к 0.069 Вт']
color = ['red', 'green', 'blue']
fontsize = 14
freqs, psd1 = get_psd(x_m11_6 + d_m11_6, Fs=fs, nfft=nfft, scale='lin')
freqs, psd2 = get_psd(x_m11_6 + d_m11_6 - y_m11_6_to_m11_6, Fs=fs, nfft=nfft, scale='lin')
freqs, psd3 = get_psd(x_m11_6 + d_m11_6 - y_m0_4_to_m11_6, Fs=fs, nfft=nfft, scale='lin')
plt.figure(31)
plt.plot(freqs + 1.8, 10 * np.log10(psd1 * alpha_11_6_dpd_off), color=color[0])
plt.plot(freqs + 1.8, 10 * np.log10(psd2 * alpha_11_6_dpd_on_appl_11_6), color=color[1])
plt.plot(freqs + 1.8, 10 * np.log10(psd3 * alpha_11_6_dpd_on_appl_0_4), color=color[2])
plt.xlabel('частота, ГГц', fontsize=fontsize)
plt.ylabel('СПМ, дБм/ГГц', fontsize=fontsize)
plt.legend(legend, fontsize=fontsize)
plt.yticks(np.arange(-100, 20, 10))
plt.ylim([-95, 5])
plt.xticks(np.arange(-2 * 0.120 + 1.8, 0.140 + 1.8, 0.040))
plt.xlim([-0.12288 + 1.8, 0.12288 + 1.8])
plt.grid()
aclr_train = aclr_fn(x_m11_6 + d_m11_6 - y_m11_6_to_m11_6, f=f, fs=fs)
aclr_test = aclr_fn(x_m11_6 + d_m11_6 - y_m0_4_to_m11_6, f=f, fs=fs)
print(f"TX DPD on. Train on -11.6 dBm. Apply to -11.6 dBm, ACLR = {aclr_train} dB")
print(f"TX DPD on. Train on -0.4 dBm. Apply to -11.6 dBm, ACLR = {aclr_test} dB")
psd1_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd1 * alpha_11_6_dpd_off)[None, :] - 30.5], axis=0).T
psd2_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd2 * alpha_11_6_dpd_on_appl_11_6)[None, :] - 30.5], axis=0).T
psd3_tmp = np.concatenate([freqs[None, :] + 1.8, 10 * np.log10(psd3 * alpha_11_6_dpd_on_appl_0_4)[None, :] - 30.5], axis=0).T
np.savetxt(os.path.join('psd', 'psd_dpd_off_0_069W.txt'), psd1_tmp)
np.savetxt(os.path.join('psd', 'psd_dpd_on_0_069Wto0_069W.txt'), psd2_tmp)
np.savetxt(os.path.join('psd', 'psd_dpd_on_0_912Wto0_069W.txt'), psd3_tmp)