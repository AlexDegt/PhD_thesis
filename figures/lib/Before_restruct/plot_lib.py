# -*- coding: utf-8 -*-
"""
Created on

@authors:
    Bakhurin Sergey (b00741681)
    Sayfullin Karim (swx959511)
    Unknown hero (python dsplib)
    
@description:
    This library provides functions for plotting:
        * abs values of the signal
        * Power spectral density
        * AM-AM, AM-PM characteristics
        * Filter frequency responces
        * Poles and zeros of digital filters
        
"""

import os
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_weight_change(Arr):
    """ """
    fig = plt.figure()
    ax = plt.subplot(111)
    for ii in range(Arr.shape[1]):
        ax.plot(Arr[:-1000,ii]/np.max(np.abs(Arr)))
        

def plot_fres(*bb, a=1, N=512, whole=True, fs=1, rad=True, plot_phase=False,
              legend=(''), fig=None, ax=None, ylim=None):
    """ Plot frequency response of the digital filter """

    if len(bb) > 1:
        plot_phase = False

    if fig is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)

    for b in bb:
        w, h = signal.freqz(b, a, N, whole, fs=fs)
        if whole:
            w = w - fs/2
            h = np.hstack((h[int(N/2):], h[:int(N/2)]))
        ax.plot(w, 20*np.log10(abs(h)))

    ax.set_title('Digital filter frequency response')
    ax.set_ylabel('Magnitude [dB]', color='b')
    ax.set_xlabel('Frequency')
    ax.set_xlim([(-fs/2), (fs/2)])
    ax.grid()
    ax.set_ylim(ylim)

    if plot_phase:
        ax2 = ax.twinx()
        angles = (np.angle(h))
        ax2.plot(w, angles/np.pi, 'g')
        ax2.set_ylabel('Angle [rad]', color='g')
        #  ax2.grid()
        ax2.axis('tight')
    plt.legend(legend)
    plt.show()



def plot(x, y, v, z, filename='', legend=(), fig=None, ax=None, figsize_x=6, figsize_y=4):
    """ Plot and save the destination folder """
    if fig is None:
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)

    if np.all(x == 0):
        ax.plot(y)
    else:
        ax.plot(x, y, v, z)

    if len(legend) > 1:
        ax.legend(legend, prop={'size': 16})
    if len(filename) > 0:
        fig.savefig(filename)
    plt.show()
    return None
    


def plot_psd(*signals, Fs=1.0, nfft=2048, filename='', legend=(), is_save=False,
             window='blackman', fig=None, ax=None, bottom_text='', top_text='',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], is_clear=True, nperseg=None, noverlap=None, y_shift=0):
    """ Plotting power spectral density """
    if fig is None:
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if is_clear:
        ax.cla()

    ax.set_xlabel('frequency')
    ax.set_xlim([-Fs/2, Fs/2])
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(ylim)
    ax.set_title('Power spectral density', fontsize=20)
    ax.grid(True)

    for iisignal in signals:
        # freqs = np.linspace(-Fs/2, Fs/2, iisignal.size)
        # plt.plot(freqs, 10*np.log10(np.fft.fftshift(np.fft.fft(iisignal))))

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, Fs, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)
        psd = 10.0*np.log10(np.fft.fftshift(psd)) + y_shift
        ax_ptr, = ax.plot(freqs, psd)
    
    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if len(legend) > 1:
        ax.legend(legend, prop={'size': 16})
    if is_save:
        fig.savefig(filename)
    plt.show()
    return ax_ptr


def plot_amam(*argv, nfig=-1, clfig=True):
    """ Plotting am-am conversion of data """
    if nfig > 0:
        plt.figure(nfig)
    else:
        plt.figure()

    if clfig:
        plt.clf()

    plt.xlabel('|x|')
    plt.ylabel('|y|')
    plt.title('am-am')
    plt.grid(True)

    if (np.mod(len(argv), 2)):
        print('Error! Length must be even.')
    else:
        for iarg in range(0, len(argv), 2):
            plt.scatter(np.abs(argv[iarg]), np.abs(argv[iarg+1]), s=0.1)
    plt.show()
    return


def plot_ampm(*argv, nfig=-1, clf=True):
    """ Plotting am-pm conversion of data """
    if nfig > 0:
        plt.figure(nfig)
    else:
        plt.figure()

    if clf:
        plt.clf()

    plt.xlabel('|x|')
    plt.ylabel('ph(y)')
    plt.title('am-pm')
    plt.grid(True)

    if (np.mod(len(argv), 2)):
        print('Error! Length must be even.')
    else:
        for iarg in range(0, len(argv), 2):
            plt.scatter(np.abs(argv[iarg]), np.angle((argv[iarg] *
                        np.conj(argv[iarg+1])), deg=True), s=0.1)
    plt.show()
    return


def plot_abs(*argv, nfig=-1, clf=True):
    """ Plotting abs value of signal """
    if nfig > 0:
        plt.figure(nfig)
    else:
        plt.figure()

    if clf:
        plt.clf()

    plt.xlabel('samples')
    plt.ylabel('|signal|')
    plt.title('abs')
    plt.grid(True)

    for iarg in range(0, len(argv)):
        leng = np.array(argv[iarg]).size
        plt.plot(np.linspace(0, leng-1, leng), np.abs(argv[iarg]))
    plt.show()
    return None

def plot_firfr(*imp_resps, Fs=1.0, nfft=1024, legend=(), \
               nfig=None, ax=None, bottom_text='', top_text='',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], clf=True):
    """ Plotting FIR frequency response """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()

    ax.set_xlabel('frequency')
    ax.set_xlim([-Fs/2, Fs/2])
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(ylim)
    ax.set_title('Frequency response', fontsize=20)
    ax.grid(True)

    for iiresp in imp_resps:
        freq_resp = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iiresp, nfft))))
        freqs = (np.arange(0, nfft)/nfft)-0.5
        ax_ptr, = ax.plot(freqs, freq_resp)
    
    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if len(legend) > 1:
        ax.legend(legend, prop={'size': 16})
    plt.show()
    return ax_ptr


def plot_spline(coeff):
    """ Plot spline amplitude conversion """

    plt.figure()
    plt.plot(np.linspace(0, 1, len(coeff)), coeff)

def plot_2dlut(*argv, dictry=[], nfig=-1, clf=True, label=[]):
    """
        Plotting 3D-graph of 2D LUT argv
        First axis - splines
        Second axis - second degree of freedom (For example RB numbers)
        dictry - dictionary used to plot second axis
    """
    dictry_len = np.size(dictry)
    nspl = np.sqrt(np.size(argv[0])).astype(int)
    size_label = np.size(label)
    if dictry_len > 0:
        X, Y = np.meshgrid(np.arange(0, nspl), np.linspace(dictry[0], dictry[dictry_len-1], nspl, dtype=int))
    else:
        X, Y = np.meshgrid(np.arange(0, nspl), np.arange(0, nspl))
    
    if nfig > 0:
        fig = plt.figure(nfig)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for iarg in range(0, len(argv)):
        curr_argv = argv[iarg]
        Z = np.zeros((nspl, nspl), dtype='float64')
        for i in range(nspl):
            Z[i, 0:nspl] = curr_argv[i*nspl:(i+1)*nspl, 0]
        ax.plot_surface(X, Y, np.abs(Z))

    if size_label == 0:
        plt.xlabel('Spline number')
        plt.ylabel('Second degree of freedom')
    elif size_label == 1:
        plt.xlabel('Spline number')
        plt.ylabel(label[0])
    elif size_label == 2:
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    elif size_label < 0 or size_label > 2:
        print("Error in function plot_2dlut\nInput number of labels should be from 0 to 2\
              Current label number: %i" %size_label)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    plt.title('2D LUT')
    return None
    


def plot_nmse(argv, sig_dict, nfig=-1, clf=True):
    """
        Plotting averaged performance being changed per time.
        argv is arr_len by sig_num array, where
        arr_len - each signal number of occurances in dataset
        sig_num - number of different signals in dataset
        sig_dict - array that contaions information about signal types
    """
    sig_num = np.shape(argv)[1]
    dict_sig_num = np.size(sig_dict)
    if sig_num != dict_sig_num:
        print("Error in function plot_nmse:\
              A number %i of signals in the input dataset doesn`t match\
              a number of elements %i in the input dictionary" %(sig_num, dict_sig_num))
        sys.exit()
    arr_len, sig_num = np.shape(argv)
    if nfig > 0:
        plt.figure(nfig)
    else:
        plt.figure()

    if clf:
        plt.clf()

    plt.xlabel('frame number')
    plt.ylabel('NMSE, dB')
    plt.title('Averaged performance per time')
    plt.grid(True)

    plt.plot(argv)
    legend = []
    for i in range(dict_sig_num):
        legend.append('RB'+str(sig_dict[i]))
    plt.legend(legend, fontsize='large')
    plt.show()
    return None

def plot_nmse_rt(*argv, ds_param, sig_dict, limits, nfig=-1, clf=True):
    """
        Plotting curves with NMSE in dynamic.
        argv - arrays of NMSE in dynamic
        ds_param - array with signal explanation.
        For example it could be the sequence of signals RB numbers
        sig_dict - array of possible signal types. For example
        array of possible RB numbers (dictionary)
        limits - array with 2 numbers: first and last
        signal number to show in graph
    """
    ds_len = np.array(argv[0]).size
    if limits[0] < 0 or limits[0] > ds_len or limits[1] < 0 or limits[1] > ds_len:
        print("Error in function plot_nmse_rt:\
              limits must be in a range from 0 to %i" %ds_len)
        sys.exit(1)
    frame_len = np.size(sig_dict)
    frame_num = int(np.floor(ds_len/frame_len))
    frame_arr = np.zeros(ds_len, dtype=int)
    label_arr = []
    for i in range(limits[0], limits[1]):
        frame_arr[i] = int(np.floor(i/frame_len))
        label_arr.append('RB'+str(ds_param[i])+'\nframe_'+str(frame_arr[i]))
    if nfig > 0:
        fig = plt.figure(nfig)
    else:
        fig = plt.figure()
    ax = fig.subplots()
    
    for iarg in range(0, len(argv)):
        curr_argv = argv[iarg]
        ax.plot(curr_argv[128*limits[0]:128*limits[1]])

    ax.set_xticks(list(np.arange(64, (limits[1]-limits[0])*128, 128)))
    ax.set_xticklabels(label_arr[0:frame_num])
    ax.grid()
    
    plt.show()
    return None


def zplane(b,a=1,auto_scale=True,size=2,detect_mult=True,tol=0.001):
    '''
    Returns
    -------
    (M,N) : tuple of zero and pole counts + plot window
    
    Unknown hero.
    '''
   
    if (isinstance(a,int) or isinstance(a,float)):
        a = [a]
    if (isinstance(b,int) or isinstance(b,float)):
        b = [b]
    M = len(b) - 1
    N = len(a) - 1
    # Plot labels if multiplicity greater than 1
    x_scale = 1.5*size
    y_scale = 1.5*size   
    x_off = 0.02
    y_off = 0.01
    #N_roots = np.array([1.0])
    if M > 0:
        N_roots = np.roots(b)
    #D_roots = np.array([1.0])
    if N > 0:
        D_roots = np.roots(a)
    if auto_scale:
        if M > 0 and N > 0:
            size = max(np.max(np.abs(N_roots)),np.max(np.abs(D_roots)))+.1
        elif M > 0:
            size = max(np.max(np.abs(N_roots)),1.0)+.1
        elif N > 0:
            size = max(1.0,np.max(np.abs(D_roots)))+.1
        else:
            size = 1.1
    plt.figure(figsize=(5,5))
    plt.axis('equal')
    r = np.linspace(0,2*np.pi,200)
    plt.plot(np.cos(r),np.sin(r),'r--')
    plt.plot([-size,size],[0,0],'k-.')
    plt.plot([0,0],[-size,size],'k-.')
    if M > 0:
        if detect_mult == True:
            N_uniq, N_mult = unique_cpx_roots(N_roots,tol=tol)
            plt.plot(np.real(N_uniq),np.imag(N_uniq),'ko',mfc='None',ms=8)
            idx_N_mult = np.nonzero(np.ravel(N_mult>1))[0]
            for k in range(len(idx_N_mult)):
                x_loc = np.real(N_uniq[idx_N_mult[k]]) + x_off*x_scale
                y_loc =np.imag(N_uniq[idx_N_mult[k]]) + y_off*y_scale
                plt.text(x_loc,y_loc,str(N_mult[idx_N_mult[k]]),ha='center',va='bottom',fontsize=10)
        else:
            plt.plot(np.real(N_roots),np.imag(N_roots),'ko',mfc='None',ms=8)                
    if N > 0:
        if detect_mult == True:
            D_uniq, D_mult=unique_cpx_roots(D_roots,tol=tol)
            plt.plot(np.real(D_uniq),np.imag(D_uniq),'kx',ms=8)
            idx_D_mult = np.nonzero(np.ravel(D_mult>1))[0]
            for k in range(len(idx_D_mult)):
                x_loc = np.real(D_uniq[idx_D_mult[k]]) + x_off*x_scale
                y_loc =np.imag(D_uniq[idx_D_mult[k]]) + y_off*y_scale
                plt.text(x_loc,y_loc,str(D_mult[idx_D_mult[k]]),ha='center',va='bottom',fontsize=10)            
        else:
            plt.plot(np.real(D_roots),np.imag(D_roots),'kx',ms=8)                
    if M - N < 0:
        plt.plot(0.0,0.0,'bo',mfc='None',ms=8)
    elif M - N > 0:
        plt.plot(0.0,0.0,'kx',ms=8)
    if abs(M - N) > 1:
        plt.text(x_off*x_scale,y_off*y_scale,str(abs(M-N)),ha='center',va='bottom',fontsize=10)        
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Pole-Zero Plot')
    #plt.grid()
    plt.axis([-size,size,-size,size])
    return M,N


def unique_cpx_roots(rlist,tol = 0.001):
    """
    
    The average of the root values is used when multiplicity 
    is greater than one.

    Mark Wickert October 2016
    """
    uniq = [rlist[0]]
    mult = [1]
    for k in range(1,len(rlist)):
        N_uniq = len(uniq)
        for m in range(N_uniq):
            if abs(rlist[k]-uniq[m]) <= tol:
                mult[m] += 1
                uniq[m] = (uniq[m]*(mult[m]-1) + rlist[k])/float(mult[m])
                break
        uniq = np.hstack((uniq,rlist[k]))
        mult = np.hstack((mult,[1]))
    return np.array(uniq), np.array(mult)