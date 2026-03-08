# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:14:31 2020

@author:
    Sayfullin Karim (swx959511)

@description:
    This library provides tools for loading data and two sync functions(not useful)

"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import support_lib as sl
from scipy.io import loadmat as loadmat
# %%

""" Signal length constants """
SAMPLEN_122_88 = 122880
IMP_RESPN_wangshw_21_11_23 = 48

""" Experiments names """
EXP_400_FRAMES_7_LEN = '400_frame_7_len_1_15_30_45_60_75_100_RB'
EXP_1000_FRAMES_20_LEN = '1000_frame_ALL_RB'
EXP_300_FRAMES_20_LEN = '300_frame_ALL_RB'
EXP_30_SIGS_50_60_54_RB = '30_sigs_50_60_54_RB'
EXP_60_SIGS_50_60_54_RB = '60_sigs_50_60_54_RB'
EXP_60_SIGS_1_10_5_RB = '60_sigs_1_10_5_RB'
EXP_120_SIGS_1_10_5_RB = '120_sigs_1_10_5_RB'
EXP_120_SIGS_90_100_96_50_60_54_RB = '120_sigs_90_100_96_50_60_54_RB'

""" Most important pathes """
DS_PARAM_400_FRAMES_7_LEN_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_400_FRAMES_7_LEN+'.npy'
DS_PARAM_1000_FRAMES_20_LEN_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_1000_FRAMES_20_LEN+'.npy'
DS_PARAM_300_FRAMES_20_LEN_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_300_FRAMES_20_LEN+'.npy'
DS_PARAM_30_SIGS_50_60_54_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_30_SIGS_50_60_54_RB+'.npy'
DS_PARAM_60_SIGS_50_60_54_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_60_SIGS_50_60_54_RB+'.npy'
DS_PARAM_60_SIGS_1_10_5_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_60_SIGS_1_10_5_RB+'.npy'
DS_PARAM_120_SIGS_1_10_5_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_120_SIGS_1_10_5_RB+'.npy'
DS_PARAM_120_SIGS_90_100_96_50_60_54_PATH = 'Useful_data/dataset_description/ds_param_'+EXP_120_SIGS_90_100_96_50_60_54_RB+'.npy'

""" Names of folders with .mat data files """
DATA_PR_122K_SAMP_NONAVER  = '122k_samples_power_ramp'
DATA_NOPR_122K_SAMP_NONAVER  = '122k_samples'
DATA_PR_122K_SAMP_AVER  = '122k_samples_power_ramp_aver'
DATA_NOPR_122K_SAMP_AVER  = '122k_samples_aver'

""" Phase jumps array names """
PJ_6000_SIGS_10_MU_6_SIGMA = 'pj_6000_sigs_10_mu_6_sigma'
PJ_NON = 'nopj'

""" Phase jumps array pathes """
PJ_6000_SIGS_10_MU_6_SIGMA_PATH = 'Useful_data/phase_jumps/'+PJ_6000_SIGS_10_MU_6_SIGMA+'.npy'

""" RB numbers of signals provided by wangshiwei 2021-11-23 """
RB_wangshw_21_11_23 = np.array([1, 5, 10, 15, 20, 25, 30, 36, 40, 45, 50, 54, 60, 64, 72, 75, 80, 90, 96, 100])
""" Some additional signal characteristic """
START_wangshw_21_11_23 = np.array(['50', '47', '45', '42', '40', '37', '35', '32', '30', '27', '25', '23', '20', '18', '14', '12', '10', '5', '2', '0'])
""" 
    PA output power (dBm) measured by powermeter. Power was measured for averaged and non-averaged data capture experiments
    Data without power ramps
    Averaged data was generated 2022-02-07
"""
PA_pwr_aver_npr_22_02_07 = np.array([5.45, 6.35, 7.25, 8.20, 9.18, 11.15, 12.10, 13.14, 14.20, 15.26, 16.34, \
                                     17.45, 18.57, 19.70, 20.77, 22.32, 23.31, 24.20, 25.13, 25.90])
""" 
    VGA power (dBm). Power was measured for averaged data capture experiments
    Data without power ramps
    Averaged data was generated 2022-02-07
"""
VGA_pwr_aver_npr_22_02_07 = np.arange(-12, 8)
""" 
    Multipathes considered by wangshiwei 00343119 to test since 2021-11-23
    These multipathes were also provided to Sayfullin Karim wx959511 
    by wangshiwei earlier.
    path_wangshw_21_11_23 contains identifiers of multipathes:
    0 - path0, 1 - path1 etc
    91 - path9_1, 123 - path12_3 etc
"""
path_wangshw_21_11_23 = np.array([0, 1, 2, 3, 4, 5])

"""
    Dictionaries used in current experiment
    Could be easily changed according to experiment
"""
CURRENT_EXP_RB = RB_wangshw_21_11_23
CURRENT_EXP_PApwr = PA_pwr_aver_npr_22_02_07
CURRENT_EXP_VGApwr = VGA_pwr_aver_npr_22_02_07
CURRENT_EXP_PATH = path_wangshw_21_11_23

MAX_RB_NUM = CURRENT_EXP_RB[CURRENT_EXP_RB.size-1]

def sym_index_transform(ds_seq, dictry, sym_to_ind):
    """
        Transforms input sequence of signal indices
        into sequence of corresponding symbols or vice versa.
        For example, such symbols could be RB numbers, multipath names etc
        Input sequence can have types: numpy.ndarray, int, str, float, 'complex128'
        
        ds_seq - input datatset sequence
        dictry - dictionary of symbols in current experiment
        sym_to_ind - flag that indicates whether to transform symbols 
        into indices or vice versa
        sym_to_ind = True - symbols to indices. ds_seq must include symbols
        sym_to_ind = False - Indices to symbols. ds_seq must include signal type indices
        
        Example: if dictry = RB_wangshw_21_11_23, sym_to_ind = True, ds_seq = np.array([5, 64, 5, 40, 100])
        then function returns np.array([1, 13, 1, 8, 19])          
    """
    if type(ds_seq) != np.ndarray:
        ds_seq = np.array([ds_seq])
    ds_seq_size = ds_seq.size
    ds_seq_res = []
    if sym_to_ind == True:
        for i in range(ds_seq_size):
            try:
                ds_seq_res.append(dictry.tolist().index(ds_seq[i]))
            except:
                print("Error in function sym_index_transform:\nSymbols dictionary doesn`t contain element %s with index %s from the input dataset sequence\n" %(ds_seq[i], i))
                sys.exit()
    if sym_to_ind == False:
        for i in range(ds_seq_size):
            try:
                ds_seq_res.append(dictry[ds_seq[i]])
            except:
                print("Error in function sym_index_transform:\nIndex %s in the input dataset sequence exceeds or equals \
                      number of elements in the symbols dictionary\n" % ds_seq[i])
                sys.exit()
    ds_seq_res = np.array(ds_seq_res)
    if ds_seq_size == 1:
        return ds_seq_res[0]
    else:
        return ds_seq_res

def sym_transform(ds_seq, dict_from, dict_to):
    """
        Transforms input sequence of symbols
        to corresponding sequence of symbols.
        For example transform RB numbers to corresponding powers of PA.
        Function works ONLY for symbols that correspond definetely to each other such that
        symbol_1[i] = symbol_2[i]
        Thus, indices of corresponding symbols MUST match
        
        ds_seq - input datatset sequence
        dict_from - dictionary of symbols to replace
        dict_to - dictionary of symbols to obtain
        
        Example: if dictry = RB_wangshw_21_11_23, sym_to_ind = True, ds_seq = np.array([5, 64, 5, 40, 100])
        then function returns np.array([1, 13, 1, 8, 19]) 
    """
    if dict_from.size != dict_to.size:
        print("Error in function sym_transform:\nSize %s of dictionary with symbols to replace doesn`t \
              equal to size %s of dictionary with symbols to replace with\n" %(dict_from.size, dict_to.size))
        sys.exit()
    ind = sym_index_transform(ds_seq, dict_from, sym_to_ind = True)
    return sym_index_transform(ind, dict_to, sym_to_ind = False)

def RB2PApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation RB number to PA power
    """
    return sym_transform(ds_seq, CURRENT_EXP_RB, CURRENT_EXP_PApwr)

def RB2VGApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation RB number to VGA power
    """
    return sym_transform(ds_seq, CURRENT_EXP_RB, CURRENT_EXP_VGApwr)

def PApwr2RB(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation PA power to RB number
    """
    return sym_transform(ds_seq, CURRENT_EXP_PApwr, CURRENT_EXP_RB)

def PApwr2VGApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation PA power to VGA power
    """
    return sym_transform(ds_seq, CURRENT_EXP_PApwr, CURRENT_EXP_VGApwr)

def VGApwr2RB(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation VGA power to RB number
    """
    return sym_transform(ds_seq, CURRENT_EXP_VGApwr, CURRENT_EXP_RB)

def VGApwr2PApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation VGA power to PA power
    """
    return sym_transform(ds_seq, CURRENT_EXP_VGApwr, CURRENT_EXP_PApwr)

def RB2ind(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation RB number to indices
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_RB, sym_to_ind=True)

def PApwr2ind(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation PA power to indices
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_PApwr, sym_to_ind=True)

def VGApwr2ind(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation VGA power to indices
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_VGApwr, sym_to_ind=True)

def ind2RB(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation indices to RB number
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_RB, sym_to_ind=False)

def ind2PApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation indices to PA power
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_PApwr, sym_to_ind=False)

def ind2VGApwr(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation indices to VGA power
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_VGApwr, sym_to_ind=False)

def path2ind(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation multipath symbols to indices
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_PATH, sym_to_ind=True)

def ind2path(ds_seq):
    """
        Simplified version of sym_transform
        for the fast transformation indices to multipath symbols
    """
    return sym_index_transform(ds_seq, CURRENT_EXP_PATH, sym_to_ind=False)

def generate_ds_param(frame_num, dictry):
    """
        Function generates dataset parameters that consist of
        frame_num frames, where each frame is random transposition 
        of signals from dictionary dictry
    """
    ds_param = []
    for i in range(frame_num):
        dictry_copy = dictry.copy()
        random.shuffle(dictry_copy)
        ds_param.extend(dictry_copy.astype(int))
    return np.array(ds_param)

def generate_HM_ds_param(frame_num, dictry_sig, dictry_path):
    """
        Function generates dataset parameters exactly for
        Hammerstein model. It consists of
        frame_num frames, where each frame is random transposition 
        of signals from dictionary dictry_sig and random transposition
        of miltipathes from dictionary dictry_path
    """
    dictry_sig_len = np.size(dictry_sig)
    dictry_path_len = np.size(dictry_path)
    if dictry_sig_len != dictry_path_len:
        print("Error in function generate_HM_ds_param:\nSignal dictionary and multipath dictionary\
              must have the same size. Now their sizes %i and %i correspondingly\n" %(dictry_sig_len, dictry_path_len))
        sys.exit()
    param_sig = generate_ds_param(frame_num, dictry_sig)
    param_path = generate_ds_param(frame_num, dictry_path)
    ds_param = np.vstack([param_sig, param_path])
    return ds_param

def fir_filter(x, w, start, y_len, is_recursive=False):
    """ 
        Perform noncausal filtering:
        x - Nx1 vector of input 
        M - fir length
        start - index for starting filtration
        end - index for ending filtration
        
        !!!Rarely using
    """
    # x = np.random.rand(1000)
    # w = np.zeros((49,1), dtype='complex128'); w[24] = 1
    # y = fir_filter(x, w, 40, 100)
    # Must return exact x numbers: [x[40], x[41] ...]

    M = len(w)
    D = int((M-1)/2)
    N = len(x)

    stn = start-D-1
    enn = start+y_len+D+1 - N
    if stn < 0:
        start = int(-1*stn)
        zer = np.zeros((start, 1))
        x = np.vstack((zer, x))
    if enn > 0:
        zer = np.zeros((enn, 1))
        x = np.vstack((x, zer))

    y = np.zeros((y_len,), dtype=x.dtype)

    if is_recursive:
        for ii in range(y_len):
            y[ii] = x[start+ii+D:start+ii-D-1:-1].T @ w
    else:
        for ii in range(y_len):
            y[ii] = x[start+ii-D:start+ii+D+1].T @ w

    return y


def compensate_plot(x, y, is_plot=False, just_delays=False):
    """ Compensate and plot abs(correlation) for given signals.
        In fact analog to compensate_delay, but specialy for FEIC theme
        
        !!!Rarely using
        Sync for input and desired signals
    """

    correl = np.abs((signal.correlate(x, y)))
    maxi = np.argmax(correl)
    leng = len(correl) + 1
    delay = int(leng/2 - maxi - 1) # supposed -1
    print('Delay is', delay, 'samples')

    if just_delays:
        return delay
    else:    
        x = x[:-delay]
        y = y[delay:]

        if is_plot:
            correl2 = np.abs(signal.correlate(x, y, method='direct'))
            leng2 = len(correl2) + 1
            plt.figure()
            # plt.xlim([int(leng/2)+5*delay, int(leng/2)-5*delay])
            w = 2
            t = np.arange(-w*delay, w*delay+1)
            plt.plot(t,(correl[int(leng/2)-w*delay: int(leng/2)+w*delay+1])/np.max(np.abs(correl)))
            plt.stem(t,(correl2[int(leng2/2)-w*delay: int(leng2/2)+w*delay+1])/np.max(np.abs(correl2)))
            peak = np.zeros(len(t)); peak[int((len(t)-1)/2)] = 1
            #plt.plot(t, peak)
            plt.xlabel('Samples')
            #plt.legend(('before', 'after'))
    
        return x, y, delay


def synchronize_all(datafolder):
    """Performs syncronization throught all data in given 
    datafolder, return and save the delay array
    
    !!!Rarely using
    Same task as compensate_plot
    """
    Delays = []
    for ii in range(len(datafolder)):
        tx = sl.import_data(datafolder[ii][1], 'txt_cmp')
        rx = sl.import_data(datafolder[ii][0], 'txt_cmp')
        txs, rxs, delay = compensate_plot(tx, rx, just_delays=True)
        Delays.append(delay)
    return Delays


def get_tx_multipathdata(
        root = r'C:\Projects\feic\2_Input_datasets\Multipath_and_highFs_luogang_07_09_20\fs122.88Mhz', 
        name = ' '):
    """ Returns tx data for 3 different Fs cases: 30, 61, 122 MHz
    
        tx start:
        /LTE20M_1RB_Start50_QPSK.mat
        /LTE20M_100RB_Start0_QPSK.mat
        
        data_l, data_m, data_h - for different sampling frequency
    """

    pow = 1
    data_h = np.zeros((2*614400, 2), dtype='complex128')
    data_l = np.zeros((int(614400/2), 2), dtype='complex128')
    data_m = np.zeros((614400, 2), dtype='complex128')

    if len(name) > 2:
        
        tmp = loadmat(root+name)
        data_m[:, 0] = tmp['x'].reshape(614400,) ** pow
        data_h[:, 0] = signal.resample_poly(tmp['x'].reshape(614400,), 2, 1) ** pow
        data_l[:, 0] = signal.decimate(tmp['x'].reshape(614400,), 2) ** pow
        
    else:
    
        tmp = loadmat(root+r'/LTE20M_1RB_Start50_QPSK.mat')
        data_m[:, 0] = tmp['x'].reshape(614400,) ** pow
        data_h[:, 0] = signal.resample_poly(tmp['x'].reshape(614400,), 2, 1) ** pow
        data_l[:, 0] = signal.decimate(tmp['x'].reshape(614400,), 2) ** pow
    
        tmp = loadmat(root+r'/LTE20M_100RB_Start0_QPSK.mat')
        data_m[:, 1] = tmp['x'].reshape(614400,) ** pow
        data_h[:, 1] = signal.resample_poly(tmp['x'].reshape(614400,), 2, 1) ** pow
        data_l[:, 1] = signal.decimate(tmp['x'].reshape(614400,), 2) ** pow
    
    return (data_l, data_m, data_h)


def return_datafolders_timedelaydata():
    """
    Returns links
    """
    data_list = []
    for ii in range(1, 3):
        for jj in range(0, 3):
            data_list.append(r'C:/Projects/feic/2_Input_datasets/high_timedelay/100RB_PAhigh_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))
    
    for ii in range(3, 7):
        for jj in range(1, 3):
            data_list.append(r'C:/Projects/feic/2_Input_datasets/high_timedelay/100RB_PAhigh_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))
    
    return data_list


def return_datafolders_svddata( ):

    data_list = [[],[],[],[],[]]

    for ii in range(1, 5):
        for jj in range(1, 5):
            data_list[0].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/1RB_PAlow/1RB_PAlow_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))

    for ii in range(1, 5):
        for jj in range(1, 5):
            data_list[1].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/1RB_PAhigh/1RB_PAhigh_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))

    for ii in range(1, 5):
        for jj in range(1, 5):
            data_list[2].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/100RB_PAlow/100RB_PAlow_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))

    for ii in range(1, 5):
        for jj in range(1, 5):
            data_list[3].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/100RB_PAhigh/100RB_PAhigh_Rxdata122.88MHz_path'+str(ii)+'_'+str(jj))

    for jj in range(1, 10):
        data_list[4].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/100RB_PAhigh_2/100RB_PAhigh_Rxdata122.88MHz_path9_'+str(jj))
    for jj in range(1, 10):
        data_list[4].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/100RB_PAhigh_2/100RB_PAhigh_Rxdata122.88MHz_path10_'+str(jj))
    for jj in range(1, 10):
        data_list[4].append(r'C:/Projects/feic/2_Input_datasets/FOR_SVD/100RB_PAhigh_2/100RB_PAhigh_Rxdata122.88MHz_path12_'+str(jj))

    return data_list
    

def return_datafolders_multipathdata(
        root=r'C:/Projects/feic/2_Input_datasets/Multipath_and_highFs_luogang_07_09_20/fs122.88Mhz'):
    """ Return all data folders (added inside func) related to 
    multipath feic topic """

    low1 = r'/1RB_PAlow_Rxdata122.88MHz_path'
    low100 = r'/100RB_PAlow_Rxdata122.88MHz_path'
    high1 = r'/1RB_PAhigh_Rxdata122.88MHz_path'
    high100 = r'/100RB_PAhigh_Rxdata122.88MHz_path'
    data_list = [[[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]]]

    for ii in range(0, 6):
        data_list[0][ii] = root+low1+str(ii+1)
        data_list[1][ii] = root+low100+str(ii+1)
        data_list[2][ii] = root+high1+str(ii+1)
        data_list[3][ii] = root+high100+str(ii+1)
    return data_list


def return_datafolder_data_with_signal():
    """ """
    
    root = r'C:/Projects/feic/2_Input_datasets/data_with_signal_22_02_21'
    RB100 = r'/100RB'
    RB1 = r'/1RB'
    name = r'/rx_data_'
    data_list = [[[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]],
            [[],[]]]
    for ii in range(12):
        data_list[ii][0] = root+RB1+name+str(ii+1)
        data_list[ii][1] = root+RB100+name+str(ii+1)
    
    return data_list
    

def retun_datafolder_different_data():
    """ """
    
    root= r'C:/Projects/feic/2_Input_datasets/Diff_dataset_26_01_21'
    rx1 = r'/LTE20M_100RB_Start0_QPSK_1'
    rx2 = r'/LTE20M_100RB_Start0_QPSK_2'
    rx3 = r'/LTE20M_50RB_Start50_QPSK_3'
    rx4 = r'/LTE20M_50RB_Start0_QPSK_4'
    tx1 = r'/rx_allrx_1'
    tx2 = r'/rx_allrx_2'
    tx3 = r'/rx_allrx_3'
    tx4 = r'/rx_allrx_4'
    
    data_list = [[[],[]],
                 [[],[]],
                 [[],[]],
                 [[],[]]]
    data_list[0][1] = root+rx1
    data_list[0][0] = root+tx1
    data_list[1][1] = root+rx2
    data_list[1][0] = root+tx2
    data_list[2][1] = root+rx3
    data_list[2][0] = root+tx3
    data_list[3][1] = root+rx4
    data_list[3][0] = root+tx4
    
    return data_list
    

def return_datafolders_refdata(nfolder):
    """Return all data folders (added inside func) related to 
    reference feic topic"""

    root = r'C:/Projects/feic\2_Input_datasets\Reference_data_luogang_14_08_20\low_complexity_FEIC_part'+str(nfolder)
    dop = r'\low_complexity_FEIC\data'
    name0 = r'\ant0.txt'
    name1 = r'\ant1.txt'
    data_list = []

    for ii in range(24):
        full_filename0 = root+dop+"\\"+str(ii+1)+name0
        full_filename1 = root+dop+"\\"+str(ii+1)+name1
        info = (full_filename0, full_filename1)
        data_list.append(info)
    return data_list


def return_datafolders_firstdata():
    """Return all data folders (added inside func) related to feic topic"""
    # Example for adapt_info:
    # 'model: Parralel Hammerstein, method: LS, FIR1 13 taps, FIR2 13 taps'
    # addition to info [weights, nmse, adapt_info, dop_info]

    #  info = (folder, data0, data1)
    # Name definition: {tx power}_{n RB}_{case n}
    keys = ['18_02_1', '22_02_1', '22_50_1', '22_50_2', '23_02_1', '23_50_1',
    '23_50_2', '23_50_3', '24_02_1', '24_50_1', '25_02_1', '25_50_1', '25_50_2']
    data_dict = {}

    f181 = r'\TxPower18dBm\2RB_start24\pre_dfir2_ant0'
    f221 = r'\TxPower22dBm\2RB_start24'
    f222 = r'\TxPower22dBm\50RB\2'
    f223 = r'\TxPower22dBm\50RB\1_4nl 13.8dB'
    f231 = r'\TxPower23dBm\2RB_start24'
    f232 = r'\TxPower23dBm\50RB\1_4nL_10.1dB_13.6'
    f233 = r'\TxPower23dBm\50RB\2_4nL_13.8dB_21'
    f234 = r'\TxPower23dBm\50RB\3_4nL_11.5dB_19.4_20Hz'
    f241 = r'\TxPower24dBm\2RB_start24\2_4nL_18.4dB_7.3'
    f242 = r'\TxPower24dBm\50RB\1_4nL_14.5dB_27.2'
    f251 = r'\TxPower25dBm\2RB_start24\1_4nL_18.7dB_11.8'
    f252 = r'\TxPower25dBm\50RB\1_4nL_13.5dB_42'
    f253 = r'\TxPower25dBm\50RB\2_4nL_14.2dB_43'
    name0 = r'\ant0.txt'
    name1 = r'\ant1.txt'
    root = r'C:/Projects/feic\2_Input_datasets\Data_from_wangshiwei_10_08_20\Desktop'

    folders = (f181, f221, f222, f223, f231, f232,
               f233, f234, f241, f242, f251, f252, f253)
    ii = 0
    for folder in folders:
        full_filename0 = root+folder+name0
        full_filename1 = root+folder+name1
        info = (full_filename0, full_filename1)
        data_dict[keys[ii]] = info
        ii = ii + 1
    return data_dict
