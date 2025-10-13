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
import support_lib as sl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
#from mpl_finance import candlestick_ohlc
# import mplfinance as mpf

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
    


def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=(), is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, noverlap=None, y_shift=0):
    """ Plotting power spectral density """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()
      
    ax.set_xlabel('frequency')
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=20)
    ax.grid(True)

    for iisignal in signals:
        # freqs = np.linspace(-Fs/2, Fs/2, iisignal.size)
        # plt.plot(freqs, 10*np.log10(np.fft.fftshift(np.fft.fft(iisignal))))

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, 1, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)*Fs
        freqs += xshift
        psd = 10.0*np.log10(np.fft.fftshift(psd)) + y_shift
        ax_ptr, = ax.plot(freqs, psd)
#        ax_ptr, = ax.plot(freqs, psd, color='tab:blue')

    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=20, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=20, ha='center', va='top')
    
    if len(legend) > 1:
        ax.legend(legend, fontsize='large')
    if is_save:
        nfig.savefig(filename)
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

def plot_firpr(*argv, legend=(), nfig=None, ax=None, figsize_x=7, figsize_y=5, clf=True):
    """ Plotting abs value of signal """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)

    if clf:
        ax.cla()

    ax.set_xlabel('frequency, rad/sample')
    ax.set_ylabel('Angle, radians')
    ax.set_title('Phase response')
    ax.grid(True)

    argv = list(argv)
    for iarg in argv:
        iarg = iarg.reshape((-1))
        w, h = signal.freqz(iarg)
        w, h = signal.freqz(iarg, worN=np.linspace(-np.pi, np.pi, 512))
        # angles = np.unwrap(np.angle(h))
        # ax.plot(w, angles)
        ax.plot(w, np.angle(h))
        
    if len(legend) > 1:
        ax.legend(legend, prop={'size': 16})
    
    plt.show()
    return None

def plot_firfr(*imp_resps, Fs=1.0, nfft=1024, legend=(), xshift=0, \
               nfig=None, ax=None, bottom_text='', top_text='',
             figsize_x=7, figsize_y=5, ylim = [-100, 10], clf=True, color=None):
    """ Plotting FIR frequency response """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()

    ax.set_xlabel('frequency, GHz')
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(ylim)
#    ax.set_title('Frequency response', fontsize=20)
    ax.grid(True)

    for iiresp in imp_resps:
        iiresp = iiresp.reshape((-1))
        freq_resp = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iiresp, nfft))))
        freqs = ((np.arange(0, nfft)/nfft)-0.5)*Fs+xshift
        if color == None:
            ax_ptr, = ax.plot(freqs, freq_resp)
        else:
            ax_ptr, = ax.plot(freqs, freq_resp, color=color)
    
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

def plot_2dlut(*argv, dictry=[], nfig=-1, clf=True, label=[], title=[]):
    """
        Plotting 3D-graph of 2D LUT argv
        First axis - splines
        Second axis - second degree of freedom (For example RB numbers)
        dictry - dictionary used to plot second axis
    """
    dictry_len = np.size(dictry)
    size_label = np.size(label)
    
    nspl = np.sqrt(np.size(argv[0])).astype(int)
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
    if title == []:
        plt.title('2D LUT')
    else:
        plt.title(title)
    return None
    
def plot_2dlut_rect(*argv, dim1, dim2, dictry=[], nfig=-1, clf=True, label=[], title=[]):
    """
        Plotting 3D-graph of 2D LUT argv
        First axis - splines
        Second axis - second degree of freedom (For example RB numbers)
        dictry - dictionary used to plot second axis
    """
    dictry_len = np.size(dictry)
    size_label = np.size(label)
    
    if nfig > 0:
        fig = plt.figure(nfig)
    else:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    if dictry_len > 0:
        X, Y = np.meshgrid(np.arange(0, dim2), np.linspace(dictry[0], dictry[dictry_len-1], dim1, dtype=int))
    else:
        X, Y = np.meshgrid(np.arange(0, dim2), np.arange(0, dim1))
    
    for iarg in range(0, len(argv)):
        curr_argv = argv[iarg]
        Z = np.zeros((dim1, dim2), dtype='float64')
        for i in range(dim1):
            Z[i, :] = curr_argv[i*dim2:(i+1)*dim2, 0]
    Z = np.reshape(curr_argv, Y.shape)
    ax.plot_surface(X, Y, np.abs(Z))
    # ax.plot_wireframe(X, Y, np.abs(Z))
    # ax.plot_trisurf(X, Y, np.abs(Z), cmap=cm.jet, linewidth=0,
    #         antialiased=False)
    # ax.plot_trisurf(np.arange(0, dim1), np.arange(0, dim1), curr_argv)
    # ax.plot_surface(X, Y, np.abs(Z), rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)    

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
    if title == []:
        plt.title('2D LUT')
    else:
        plt.title(title)
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

def plot_nmse_rt(*argv, ds_param, sig_dict, limits=[], nfig=-1, frame_len=-1, nmse_sig_len=128, clf=True, 
                 legend=[], title='Dynamic NMSE', xlabel='time', ylabel='NMSE, dB', showframe=False):
    """
        Plotting curves with NMSE in dynamic.
        argv - arrays of NMSE in dynamic
        ds_param - array with signal explanation.
        For example it could be the sequence of signals RB numbers
        sig_dict - array of possible signal types. For example
        array of possible RB numbers (dictionary)
        limits - array with 2 numbers: first and last
        signal number to show in graph
        
        Example of use:
        ds_param = np.array(list(sl.txt2dict(r'..\dataset\data\0\ds_param.txt', int).values())[0]).T.tolist()
        ds_param = np.array(ds_param)[0, :]
        sig_dict = feic.RB_wangshw_21_11_23
        pl.plot_nmse_rt(nmse_rt, ds_param=ds_param, sig_dict=sig_dict)
    """   
    ds_len = 0
    for iarg in range(0, len(argv)):
        curr_argv = argv[iarg]
        curr_argv_size = np.size(curr_argv)
        if ds_len < curr_argv_size:
            ds_len = curr_argv_size
            
    if limits != []:
        if limits[0] < 0 or limits[0] > ds_len or limits[1] < 0 or limits[1] > ds_len:
            print("Error in function plot_nmse_rt:\
                  limits must be in a range from 0 to %i" %ds_len)
            sys.exit(1)
        
    if frame_len == -1:
        frame_len = np.size(sig_dict)
        
    sig_num = int(np.floor(ds_len/nmse_sig_len))
    frame_arr = np.zeros(ds_len, dtype=int)
    label_arr = []

    limits.append(0)
    limits.append(sig_num)
    
    for i in range(limits[0], limits[1]):
        frame_arr[i] = int(np.floor(i/frame_len))
        if showframe:
            label_arr.append('RB'+str(ds_param[i])+'\nframe_'+str(frame_arr[i]))
        else:
            label_arr.append('RB'+str(ds_param[i]))
    if nfig > 0:
        fig = plt.figure(nfig)
    else:
        fig = plt.figure()
    ax = fig.subplots()
    
    for iarg in range(0, len(argv)):
        curr_argv = argv[iarg]
        ax.plot(curr_argv[128*limits[0]:128*limits[1]])

    ax.set_xticks(list(np.arange(64, (limits[1]-limits[0])*128, 128)))
    ax.set_xticklabels(label_arr[0:sig_num])
    ax.minorticks_on()
    ax.grid(which='major', color='k', linewidth=1)
    ax.grid(which='minor', color = 'k', linestyle = ':', linewidth=1)
    
    fontsize = 15
    
    if legend != []:
        ax.legend(legend, fontsize=fontsize)
    
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)        
    
    plt.show()
    return None

def calc_candle(*argv, dictry, perct=0.8, in_db = True, block_num=128, block_len=960, start=[]):
    """ 
        Calculating arrays for candle diagram
        
        Examples of function use:
        pl.calc_candle(nmse_newton, dictry=feic.RB_wangshw_21_11_23)
        pl.calc_candle(e_sqr_rt_mem, dictry=feic.RB_wangshw_21_11_23, in_db=False)
    """    
    num_of_rb = dictry.size 
    argv_len = len(argv)
    
    if start == []: start.extend(np.zeros(argv_len))
    if type(start) != list:
        print('Error:')
        print('Parameter \'start\' must have type \'list\'')
        sys.exit()
    if np.size(start) != argv_len:
        print('Error:')
        print('Parameter \'start\' must consist of %i elements'%argv_len)
        sys.exit()
            
    stat=[]
    
    for iarg in range(0, argv_len):
        ds_len = np.shape(argv[iarg])[1]*np.shape(argv[iarg])[0]
        if ds_len < start[iarg]:
            print('Error:')
            print('Slot number in dataset is lower than start slot number. Decrease parameter \'start\'')
            sys.exit()
        max_arr_whole = np.zeros(num_of_rb)
        min_arr_whole = np.zeros(num_of_rb)
        accuracy_high_arr_whole = np.zeros(num_of_rb)
        accuracy_low_arr_whole = np.zeros(num_of_rb)
        curr_argv = np.array(argv[iarg])
        ds_len = curr_argv.shape[0]*curr_argv.shape[1]
        for i in range(0, num_of_rb):
            curr_arr = curr_argv[i, int(np.ceil(((start[iarg])/num_of_rb))):, :].reshape(block_num*int(np.ceil((((ds_len-start[iarg])/num_of_rb)))),)
            if in_db == True:
                curr_arr = 10**(curr_arr/10)
                # print(curr_arr)
            x_min, x_max, x_mean, _ = sl.search_eff_int(curr_arr, perct=perct)
            max_arr_whole[i] = np.max(curr_arr)
            min_arr_whole[i] = np.min(curr_arr)
            accuracy_high_arr_whole[i] = x_max
            accuracy_low_arr_whole[i] = x_min
            
            max_arr_whole[i] = 10*np.log10(max_arr_whole[i])
            min_arr_whole[i] = 10*np.log10(min_arr_whole[i])
            accuracy_high_arr_whole[i] = 10*np.log10(accuracy_high_arr_whole[i])
            accuracy_low_arr_whole[i] = 10*np.log10(accuracy_low_arr_whole[i])
        stat.append([accuracy_high_arr_whole, max_arr_whole, min_arr_whole, accuracy_low_arr_whole])
    
    return stat

#def plot_candle(*argv, dictry, stat=[], perct=0.8, in_db=True, block_num=128, block_len=960, start=[], 
#                nfig=-1, clf=True, grid=True, legend=[], color=None, width_lim=[1.0, 2.5], mmwidth_lim=[1.0, 4.0],
#                width_arr=[], mmwidth_arr=[], fontsize=15):
#    """ 
#        Plotting candle diagram
#        
#        Examples of function use:
#        pl.plot_candle(nmse_newton, dictry=feic.RB_wangshw_21_11_23)
#        pl.plot_candle(10*np.log10(e_sqr_rt_mem), nmse_newton, dictry=feic.RB_wangshw_21_11_23, 
#                       legend=['LMS-LMS Min-max interval', 'LMS-LMS 80% points interval', 
#                       'Damped Newton Min-max interval', 'Damped Newton 80% points interval'])
#        pl.plot_candle([], stat=stat, dictry=feic.RB_wangshw_21_11_23, 
#                       legend=['LMS-LMS Min-max interval', 'LMS-LMS 80% points interval', 
#                       'Damped Newton Min-max interval', 'Damped Newton 80% points interval'])
#    """ 
#    
#    num_of_rb = dictry.size
#    
#    if nfig > 0:
#        ax = plt.subplot(nfig)
#    else:
#        ax = plt.subplot()
#    
#    color_arr = np.array(['black', 'red', 'cyan', 'purple', 'blue'])
#    color_arr_len = color_arr.size
#    
#    argv_len_stat = np.shape(stat)[0]
#    if argv == ([],): argv_len_argv = 0 
#    else: argv_len_argv = len(argv)
#    argv_len_gen = argv_len_stat+argv_len_argv
#    if width_arr == []:
#        width_arr = np.linspace(width_lim[0], width_lim[1], argv_len_gen)
#    else:
#        width_arr = np.array(width_arr)
#    if mmwidth_arr == []:
#        mmwidth_arr = np.linspace(mmwidth_lim[0], mmwidth_lim[1], argv_len_gen)
#    else:
#        mmwidth_arr = np.array(mmwidth_arr)
#    if argv_len_gen == 1: 
#        width_arr = np.array([width_lim[1]])
#        mmwidth_arr = np.array([mmwidth_lim[1]])
#    
#    if stat != []:
#       for iarg in range(0, argv_len_stat):
#           ohlc = []
#           for i in range(0, num_of_rb):
#               tmp = dictry[i], stat[iarg][0][i] ,stat[iarg][1][i], stat[iarg][2][i], stat[iarg][3][i]
#               ohlc.append(tmp)     
#           if iarg < color_arr_len:
#               colordown = color_arr[iarg]
#           else:
#               colordown = color_arr[color_arr_len - 1]      
#           candlestick_ohlc(ax, ohlc, width=width_arr[argv_len_gen-iarg-1], 
#                            mmwidth=mmwidth_arr[argv_len_gen-iarg-1], colordown=colordown, alpha=0.9)     
#    if argv != ([],):
#        stat = calc_candle(*argv, dictry=dictry, perct=perct, in_db=in_db, start=start) 
#        for iarg in range(0, argv_len_argv):
#            ohlc = []
#            for i in range(0, num_of_rb):
#                tmp = dictry[i], stat[iarg][0][i] ,stat[iarg][1][i], stat[iarg][2][i], stat[iarg][3][i]
#                ohlc.append(tmp)     
#            if iarg < color_arr_len:
#                colordown = color_arr[argv_len_stat+iarg]
#            else:
#                colordown = color_arr[color_arr_len - 1]
#            candlestick_ohlc(ax, ohlc, width=width_arr[argv_len_argv-iarg-1],
#                             mmwidth=mmwidth_arr[argv_len_argv-iarg-1], colordown=colordown, alpha=0.9)
#     
#    if legend == []:
#        ax.legend(['Min-max interval', '%d%% points interval'%(perct*100)], fontsize=15)
#    else:
#        ax.legend(legend, fontsize=fontsize)
#    
#    leg = ax.get_legend()
#    
#    if color == None:
#        for i in range(argv_len_gen):     
#            leg.legendHandles[2*i].set_color(color_arr[i])
#            leg.legendHandles[2*i+1].set_color(color_arr[i])
#    else:
#       for i in range(argv_len_gen):     
#           leg.legendHandles[2*i].set_color(color)
#           leg.legendHandles[2*i+1].set_color(color) 
#        
#    ax.set_xticks(dictry)
#    ax.set_xlabel('RB number', fontsize=fontsize)
#    ax.set_ylabel('NMSE, dB', fontsize=fontsize)
#    ax.set_title('Performance statistics', fontsize=fontsize)
#    
#    return None

def calc_statsig(*argv, dictry, rb_num, perct=0.8, block_num=128, block_len=960, start=0):
    """ 
        Calculation of certain RB number statistics among whole dataset
        
        Examples of function use:
        pl.calc_statsig(e_sqr_rt_mem, dictry=feic.RB_wangshw_21_11_23, rb_num=1)
    """  
    rb_ind = sl.feic.RB2ind(rb_num) 
    num_of_rb = dictry.size
    argv_len = len(argv)   
    stat=[]
    
    for iarg in range(0, argv_len):
        mean_arr = np.zeros((num_of_rb, block_num), dtype=float)
        max_arr = block_len*np.zeros((num_of_rb, block_num), dtype=float)
        min_arr = np.ones((num_of_rb, block_num), dtype=float)
        accuracy_arr = np.zeros((num_of_rb, 2, block_num), dtype=float)
        curr_argv = np.array(argv[iarg])
        for i in range(0, num_of_rb):
            for j in range(0, block_num):
                x_min, x_max, x_mean, _ = sl.search_eff_int(curr_argv[i, int(np.ceil(((start)/num_of_rb))):, j], perct=perct)
                accuracy_arr[i, 0, j] = x_max
                accuracy_arr[i, 1, j] = x_min
                mean_arr[i, j] = x_mean
                max_arr[i, j] = np.max(curr_argv[i, int(np.ceil(((start)/num_of_rb))):, j])
                min_arr[i, j] = np.min(curr_argv[i, int(np.ceil(((start)/num_of_rb))):, j])
                
        max_arr_rt = 10*np.log10(max_arr[rb_ind, :])
        min_arr_rt = 10*np.log10(min_arr[rb_ind, :])
        mean_arr_rt = 10*np.log10(mean_arr[rb_ind, :])
        accur_high_arr_rt = 10*np.log10(accuracy_arr[rb_ind, 0, :])
        accur_low_arr_rt = 10*np.log10(accuracy_arr[rb_ind, 1, :])
        
        stat.append([accur_high_arr_rt, max_arr_rt, mean_arr_rt, min_arr_rt, accur_low_arr_rt])
    
    return stat

def plot_statsig(*argv, dictry, rb_num, stat=[], perct=0.8, block_num=128, block_len=960, start=0, 
                nfig=-1, clf=True, grid=True, legend=[]):
    """ 
        Plotting certain RB number statistics among whole dataset
        
        Examples of function use:
        pl.plot_statsig(e_sqr_rt_mem, dictry=feic.RB_wangshw_21_11_23, rb_num=1)
    """
    
    argv_len = len(argv)
    if nfig > 0:
        plt.figure(nfig)
    else:
        plt.figure()

    if clf:
        plt.clf()
    
    if stat == []:
        stat = calc_statsig(*argv, dictry=dictry, rb_num=rb_num, perct=perct, block_num=block_num, block_len=block_len, start=start)
    
    for iarg in range(0, argv_len):
        accur_high_arr_rt, max_arr_rt, mean_arr_rt, min_arr_rt, accur_low_arr_rt = stat[iarg]
    
        plt.fill_between(np.arange(block_num), max_arr_rt, min_arr_rt, alpha=0.3, color='y')
        plt.plot(mean_arr_rt, color='black')
        plt.fill_between(np.arange(block_num), accur_high_arr_rt, accur_low_arr_rt, alpha=0.3, color='red')
        
    plt.xlabel('time')
    plt.ylabel('NMSE, dB')
    if legend == []:
        plt.legend(['Min-Max interval', 'Mean', 'Interval with 80% points'], fontsize=15)    
    else:
        plt.legend(legend, fontsize=15)
    plt.title('Performance statistics', fontsize=15)
    plt.grid(grid)
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

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts