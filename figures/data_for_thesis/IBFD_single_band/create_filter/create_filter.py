import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

import copy

sys.path.insert(0, "../../lib")
import plot_lib as pl
import support_lib as sl
import scipy.signal as signal
import scipy
from time import perf_counter

fontsize = 13

taps1 = signal.firls(numtaps=51, bands=[0, 0.005, 0.05, 0.5], desired=[0.7, 0.7, 0, 0], fs=1) + 1j*0
taps2 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.1, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0
taps3 = signal.firls(numtaps=51, bands=[0, 0.06, 0.0601, 0.5], desired=[0.7, 0.7, 0, 0], fs=1) + 1j*0
taps4 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.07, 0.5], desired=[1, 1, 0, 0], fs=1) + 1j*0
taps5 = signal.firls(numtaps=51, bands=[0, 0.04, 0.041, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0
taps6 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.1, 0.5], desired=[0.8, 0.8, 0, 0], fs=1) + 1j*0
taps7 = signal.firls(numtaps=51, bands=[0, 0.06, 0.0601, 0.5], desired=[0.8, 0.8, 0, 0], fs=1) + 1j*0
taps8 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.075, 0.5], desired=[1, 1, 0, 0], fs=1) + 1j*0
taps9 = signal.firls(numtaps=51, bands=[0, 0.07, 0.0701, 0.5], desired=[0.7, 0.7, 0.1, 0], fs=1) + 1j*0
taps10 = 1j*signal.firls(numtaps=51, bands=[0, 0.04, 0.041, 0.5], desired=[0.77, 0.77, 0, 0], fs=1) + 1j*0
taps11 = signal.firls(numtaps=51, bands=[0, 0.005, 0.075, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0
taps12 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.2, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0

taps13 = signal.firls(numtaps=51, bands=[0, 0.047, 0.04701, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0
taps14 = 1j*signal.firls(numtaps=51, bands=[0, 0.005, 0.2, 0.5], desired=[0.9, 0.9, 0, 0], fs=1) + 1j*0

n_shift1 = -9
n_shift2 = -5
n_shift3 = -13
n_shift4 = -17
n_shift5 = -21
n_shift6 = -25
n_shift7 = -2
n_shift8 = 3.15
n_shift9 = 6
n_shift10 = 11
n_shift11 = 15.2
n_shift12 = 20
n_shift13 = 15.15
n_shift14 = 0
shift1 = np.exp(-1j*2*n_shift1*np.pi*np.arange(taps1.size)/taps1.size)
shift2 = np.exp(-1j*2*n_shift2*np.pi*np.arange(taps2.size)/taps2.size)
shift3 = np.exp(-1j*2*n_shift3*np.pi*np.arange(taps3.size)/taps3.size)
shift4 = np.exp(-1j*2*n_shift4*np.pi*np.arange(taps4.size)/taps4.size)
shift5 = np.exp(-1j*2*n_shift5*np.pi*np.arange(taps5.size)/taps5.size)
shift6 = np.exp(-1j*2*n_shift6*np.pi*np.arange(taps6.size)/taps6.size)
shift7 = np.exp(-1j*2*n_shift7*np.pi*np.arange(taps7.size)/taps7.size)
shift8 = np.exp(-1j*2*n_shift8*np.pi*np.arange(taps8.size)/taps8.size)
shift9 = np.exp(-1j*2*n_shift9*np.pi*np.arange(taps9.size)/taps9.size)
shift10 = np.exp(-1j*2*n_shift10*np.pi*np.arange(taps10.size)/taps10.size)
shift11 = np.exp(-1j*2*n_shift11*np.pi*np.arange(taps11.size)/taps11.size)
shift12 = np.exp(-1j*2*n_shift12*np.pi*np.arange(taps12.size)/taps12.size)
shift13 = np.exp(-1j*2*n_shift13*np.pi*np.arange(taps13.size)/taps13.size)
shift14 = np.exp(-1j*2*n_shift14*np.pi*np.arange(taps14.size)/taps14.size)
taps1 *= shift1
taps2 *= shift2
taps3 *= shift3
taps4 *= shift4
taps5 *= shift5
taps6 *= shift6
taps7 *= shift7
taps8 *= shift8
taps9 *= shift9
taps10 *= shift10
taps11 *= shift11
taps12 *= shift12
taps13 *= shift13
taps14 *= shift14

taps = 1*(taps1 + taps2 + taps3 + taps4 + taps5 + taps6 + taps7 + taps8 + taps9 + taps10 + taps1 + taps12) + taps13 + 0*taps14

nfft = 1024
xlim = [1.8 - 0.06144, 1.8 + 0.06144]
freqs = np.linspace(xlim[0], xlim[1], nfft)

pl.plot_firfr(taps, freqs=freqs, nfft=nfft)

plt.xlabel("Частота, ГГц", fontsize=fontsize)
plt.ylabel("АЧХ, дБ", fontsize=fontsize)
plt.ylim([-10, 5])
plt.xlim(xlim)

np.save("channel_filter.npy", taps)