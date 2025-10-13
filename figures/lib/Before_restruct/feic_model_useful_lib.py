# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:17:41 2020

@author:
    Sayfullin Karim (swx959511)

@description:
    This library provides models for modelling feic compensator. Such as:
        * Parallel Hammerstein with different number of branches.
        * Hammerstein with (polynomial nonlinearity and lut nonlinearity)

    There are different types of adaptation algorithms:
        * LS
        * WRLS
        * DCD
        * LMS
        * Newton

"""
import sys
sys.path.insert(1, '../lib/')
import numpy as np
import scipy.signal as signal
from scipy.io import loadmat as loadmat
from scipy.linalg import pinv as pinv
import matplotlib.pyplot as plt

import plot_lib as pl
import support_lib as sl
import dpd_lib as dpd
import dpd_Sergey_lib as dpd_sergey
import model_lib as dpd_model
import feic_lib as feic

# %%

def Hammerstein_LMS_DCD_parallel(x, d, w_fir, w_lut, pwr=1, mu_lut = 0.0001, lam=0.9995,
                                 Nu=16, Mb=64, nepoch=20, sAcc=100, sWait=100, nReppts=80,
                                 c_leakage=1, c_momentum=0, start_state='acc',
                                 nmse_newton=-100, accuracy=1, is_return_weights=False,
                                 is_plot=False, is_write=False, is_retsig=True):
    """ Hammerstein with LMS adaptation for LUT and WRLSDCD for FIR 
        x - input signal(not squared)
        d - desired signal
        w_fir - filter weights
        w_lut - lut weights
        pwr - power for multiplier
        mu - LMS-DCD LUT parameter
        lam - decay param (RLS)
        Nu - precision (DCD)
        Mb - number of cycles per solving (DCD)
        nepoch - number of epochs for iteration
        nReppts - number of reper points per each iteration (typical = 80)
        sAcc - number of samples for updating LUT 
        sWait - number of smaples for backpropagating error (usually 2*fir_delay or 1)
        c_leakage - decay coefficient
        c_momentum - momentum coefficient (if 0, then no momentum)
        start_state - initial state for state machine
        nmse_newton - precalculated best results to compare with
        accuracy - precision for results [dB]
    """

    N = len(x)
    M = w_fir.shape[0]
    nspl = w_lut.shape[0]
    numToSave = int(N/nReppts)
    
    Ww  = []
    DpD = []
    NmsE = []
    
    Nmse = []
    err = np.zeros((N*nepoch,), dtype='complex128')
    fir_acc = np.zeros((N*nepoch, ))
    lut_acc = np.zeros((N*nepoch, ))
    
    #if c_momentum == 0:
    #    None
    #else:
    dw_lut1 = np.zeros((N,), dtype='complex128')
    dw_lut2 = np.zeros((N,), dtype='complex128')
    
    if is_return_weights:
        Wfir = np.zeros((N*nepoch, M), dtype='complex128')
        Wlut = np.zeros((N*nepoch, nspl), dtype='complex128')

    r = np.zeros((M,), dtype='complex128')    
    sigma = np.ones((M,), dtype='complex128')*0.0001  # Defines initial corr matrix
    R = np.diag(sigma)
    delay_fir = int((M-1)/2)
    u = np.zeros((M, 1), dtype='complex128')
    bp_buf1 = np.zeros((M, 1), dtype='complex128')
    w_dop1 = w_fir

    xa = np.abs(x)*nspl
    addr = (np.floor(xa)).astype('int')
    dx = xa - addr
   
    for jj in range(nepoch):
        cnt = 0
        curr_state = start_state
        for ii in range(N):
            
            m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
            f = m * (x[ii])**pwr
            
            u = np.vstack((f, u[:-1])) 
            y = np.matmul(np.transpose(w_fir), u)
            e = d[ii] - y
            err[jj*N + ii] = e
            e1 = d[ii] - np.matmul(np.transpose(w_dop1), u)
            bp_buf1 = np.vstack((e1, bp_buf1[:-1]))
            e_bp1 = np.matmul(np.transpose(np.conj(np.flip(w_fir))), bp_buf1)

            R = lam*R + np.matmul(np.conj(u), np.transpose(u))
            b = lam*r + (e*np.conj(u)).reshape((M,))
            dw, r = complex_leading_DCD(R, b, M, Nu, Mb)

            if ii > delay_fir*2:
    
                w_fir = w_fir + dw
                fir_acc[jj*N + ii] = np.sum(np.abs(dw))
                
                if curr_state == 'acc':
                    w_dop1 = w_fir
                    #if c_momentum == 0:
                    #    w_lut[addr[ii-delay_fir*2]] = c_leakage*w_lut[addr[ii-delay_fir*2]] + mu_lut*(1-dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                    #    w_lut[addr[ii-delay_fir*2]+1] = c_leakage*w_lut[addr[ii-delay_fir*2]+1] + mu_lut*(dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                    #else:
                    dw_lut1[ii] = c_momentum*dw_lut1[ii-1] - mu_lut*(1-dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                    dw_lut2[ii] = c_momentum*dw_lut2[ii-1] - mu_lut*(dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                    w_lut[addr[ii-delay_fir*2]] = c_leakage*w_lut[addr[ii-delay_fir*2]] - dw_lut1[ii]
                    w_lut[addr[ii-delay_fir*2]+1] = c_leakage*w_lut[addr[ii-delay_fir*2]+1] - dw_lut2[ii]
                    
                    lut_acc[jj*N + ii] = np.abs(dw_lut1[ii]) + np.abs(dw_lut2[ii])
                    
                    cnt += 1
                    if cnt >= sAcc:
                        curr_state = 'acc'
                        cnt = 0
                        if sWait == 0:
                            curr_state = 'acc'
                else:
                    if curr_state == 'wit':
                        cnt += 1
                        if cnt >= sWait:
                            curr_state = 'acc'
                            cnt = 0
                    else:
                        curr_state = 'wit'
                        cnt = 0 
                        
            if is_return_weights:
                Wfir[jj*N + ii, :] = w_fir.reshape(-1,)
                Wlut[jj*N + ii, :] = w_lut.reshape(-1,)
            
            
            '''
            is_time_to_save = not(ii % numToSave)
            if is_time_to_save:
                g = np.zeros((N,), dtype='complex128')
                for ii in range(N):
                    m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
                    g[ii] = m*(x[ii])**pwr
                dpd_sig = dpd.fir_filter(g, w_fir)
                dpd_sig.resize(N,)
                nmse = sl.nmse(x, d-dpd_sig)
                Nmse.append(nmse)
            '''
        # One of the stop criteria
        #is_close_to_optimum = np.abs(nmse_newton - nmse) < accuracy
        #if is_close_to_optimum:
        #    print('Converges faster than needed!')
        #    break
        
        g = np.zeros((N,), dtype='complex128')
        for ii in range(N):
            m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
            g[ii] = m*(x[ii])**pwr
        dpd_sig = dpd.fir_filter(g, w_fir)
        dpd_sig.resize(N,)
        NmsE.append(sl.nmse(x, d-dpd_sig))
        if not(jj % 2):
            DpD.append(dpd_sig)
        
        if jj == 800:
            mu_lut /= 2
        if jj == 950:
            mu_lut /= 3
        if jj == 1250:
            mu_lut /= 5
        if jj == 1400:
            mu_lut /= 10

    info = ('''Model: Hammerstein,
       method: LMS-DCD with parallel strategy of adaptation,
       mu_lut(initial)''', str(mu_lut), 'lam', str(lam), 'Nu', str(Nu),
       '''FIR''', M, 'taps')
    w_itog = np.concatenate((w_fir, w_lut), axis=0)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig, legend=('desired', 'compensator', 'error'))
    if is_write:
        print('NMSE after finding solution:', nmse)
    if is_retsig:
        None
    else:
        dpd_sig = 0
    if is_return_weights:
        return dpd_sig, err, w_itog, np.array(Nmse), info, fir_acc , lut_acc, Wfir, Wlut, Ww, DpD, NmsE
    else:
        return dpd_sig, err, w_itog, np.array(Nmse), info, fir_acc, lut_acc, Ww, DpD, NmsE



def Hammerstein_LMS_DCD_serial(x, d, w_fir, w_lut, pwr=1, mu_lut=0.001, lam=0.9995, 
                                 Nu=16, Mb=64, nepoch=20, sFIR = 500, sWait = 100, sLUT = 500,
                                 nReppts=80, c_leakage=1, c_momentum=0, start_state='lut',
                                 nmse_newton=-100, accuracy=1, is_return_weights=False,
                                 is_plot=False, is_write=False, is_retsig=True):
    """ Hammerstein with LMS adaptation for LUT and WRLSDCD for FIR. Sequential
         or serial adaptation.
        x - input signal(not squared)
        d - desired signal
        w_fir - filter weights
        w_lut - lut weights
        pwr - power for multiplier
        mu_lut - LMS-DCD LUT parameter
        lam - decay param (RLS)
        Nu - precision (DCD)
        Mb - number of cycles per solving (DCD)
        nepoch - number of epochs for iteration
        sFIR - number of samples for filter adaptation
        sWait - number of samples for backpropagation (usually 1 or 2*filter_delay)
        sLUT - number of samples for LUT adaptation 
        c_leakage - decay coefficient
        c_momentum - momentum coefficient (if 0, then no momentum)
        start_state - initial state for state machine
        nmse_newton - precalculated best results to compare with
        accuracy - precision for results [dB]
    """

    N = len(x)
    M = w_fir.shape[0]
    nspl = w_lut.shape[0]
    numToSave = int(N/nReppts)

    Nmse = []
    err = np.zeros((N*nepoch,), dtype='complex128')
    
    if c_momentum == 0:
        None
    else:
        dw_lut1 = np.zeros((N,), dtype='complex128')
        dw_lut2 = np.zeros((N,), dtype='complex128')
    if is_return_weights:
        Wfir = np.zeros((N*nepoch, M), dtype='complex128')
        Wlut = np.zeros((N*nepoch, nspl), dtype='complex128')
    
    r = np.zeros((M,), dtype='complex128')    
    sigma = np.ones((M,), dtype='complex128')*0.0001 # This multiplier determines initial matrix
    R = np.diag(sigma)
    delay_fir = int((M-1)/2)
    u = np.zeros((M, 1), dtype='complex128')
    bp_buf1 = np.zeros((M, 1), dtype='complex128')

    xa = np.abs(x)*nspl
    addr = (np.floor(xa)).astype('int')
    dx = xa - addr
    
    for jj in range(nepoch):
        cnt = 0
        curr_state = start_state
    
        for ii in range(N):
    
            m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
            f = m * (x[ii])**pwr
    
            u = np.vstack((f, u[:-1])) 
            y = np.matmul(np.transpose(w_fir), u)
            e = d[ii] - y 
            err[jj*N + ii] = e

            bp_buf1 = np.vstack((e, bp_buf1[:-1]))
            e_bp1 = np.matmul(np.transpose(np.conj(np.flip(w_fir))), bp_buf1)
            
            R = lam*R + np.matmul(np.conj(u), np.transpose(u))
            b = lam*r + (e*np.conj(u)).reshape((M,))
            dw, r = complex_leading_DCD(R, b, M, Nu, Mb)
            
            if ii > delay_fir*2:
                if curr_state == 'fir':
                    w_fir = w_fir + dw
                    cnt += 1
                    if cnt >= sFIR:
                        curr_state = 'lut'  ##########
                        cnt = 0
                else:
                    if curr_state == 'wit':
                        cnt += 1
                        if cnt >= sWait:
                            curr_state = 'lut'
                            cnt = 0    
                    else:
                        if curr_state == 'lut':
                            if c_momentum == 0:
                                w_lut[addr[ii-delay_fir*2]] = c_leakage*w_lut[addr[ii-delay_fir*2]] + mu_lut*(1-dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                                w_lut[addr[ii-delay_fir*2]+1] = c_leakage*w_lut[addr[ii-delay_fir*2]+1] + mu_lut*(dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                            else:
                                dw_lut1[ii] = c_momentum*dw_lut1[ii-1] - mu_lut*(1-dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                                dw_lut2[ii] = c_momentum*dw_lut2[ii-1] - mu_lut*(dx[ii-delay_fir*2])*e_bp1*np.conj(x[ii-delay_fir*2]**pwr)*(1/np.abs(x[ii-delay_fir*2])**2)
                                w_lut[addr[ii-delay_fir*2]] = c_leakage*w_lut[addr[ii-delay_fir*2]] - dw_lut1[ii]
                                w_lut[addr[ii-delay_fir*2]+1] = c_leakage*w_lut[addr[ii-delay_fir*2]+1] - dw_lut2[ii]
                            cnt += 1
                            if cnt >= sLUT:
                                curr_state = 'fir' ###############
                                cnt = 0  
                        else:
                            curr_state = 'fir'
                            cnt = 0
            
            if is_return_weights:
                Wfir[jj*N + ii, :] = w_fir.reshape(-1,)
                Wlut[jj*N + ii, :] = w_lut.reshape(-1,)
            '''
            is_time_to_save = not(ii % numToSave)
            if is_time_to_save:
                g = np.zeros((N,), dtype='complex128')
                for ii in range(N):
                    m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
                    g[ii] = m*(x[ii])**pwr
                dpd_sig = dpd.fir_filter(g, w_fir)
                dpd_sig.resize(N,)
                nmse = sl.nmse(x, d-dpd_sig)
                Nmse.append(nmse)
            '''
        g = np.zeros((N,), dtype='complex128')
        for ii in range(N):
            m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
            g[ii] = m*(x[ii])**pwr
        dpd_sig = dpd.fir_filter(g, w_fir)
        dpd_sig.resize(N,)
        nmse = sl.nmse(x, d-dpd_sig)
        Nmse.append(nmse)
        # One of the stop criteria
        #is_close_to_optimum = np.abs(nmse_newton - nmse) < accuracy
        #if is_close_to_optimum:
        #    print('Converges faster than needed!')
        #    break
        
        if jj == 25:
            mu_lut /= 4
        if jj == 34:
            mu_lut /= 4
        if jj == 75:
            mu_lut /= 4
        if jj == 90:
            mu_lut /= 10
        
    info = ('''Model: Hammerstein, method: LMS-DCD,
            with sequential strategy of adaptation,
            FIR''', M, 'taps', 'lut', nspl)
    w_itog = np.concatenate((w_fir, w_lut), axis=0)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig, legend=('desired', 'compensator', 'error'))
    if is_write:
        print('NMSE after finding solution:', nmse)
    if is_retsig:
        None
    else:
        dpd_sig = 0
    if is_return_weights:
        return dpd_sig, err, w_itog, np.array(Nmse), info, Wfir, Wlut
    else:
        return dpd_sig, err, w_itog, np.array(Nmse), info
    
# %% For convenience

def Hammerstein_out(x, w_lut, w_fir, pwr):
    
    N = len(x)
    y = np.zeros((N,), dtype='complex128')

    xa = np.abs(x)*w_lut.shape[0]
    addr = (np.floor(xa)).astype('int')
    dx = xa - addr

    for ii in range(N):
        m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
        y[ii] = m*(x[ii])**pwr
    
    return dpd.fir_filter(y, w_fir, is_noncausal=False)


# %% Newton algorithm

def Newton_solver(calc_out, solve, 
                  x, d, w_full, nspl, M, pwr,
                  mu):
    """ Calculate Iterative Newton solution for given structure
        calc_out - function for calculating model output
        solve - function for solving full Hessian and gradient returns delta weights
        x - input signal
        d - desired signal (Delay between x and d must be zero!)
        w_full - combined vector for LUT and FIR
        nspl - number of splines in LUT
        M - number of FIR coefficients
        pwr - power for LUT output: output = lut(|x|) * (x**pwr)
        mu - initial value for stepsize
    """
    
    if w_full.shape[0] == 1:
        w_fir = np.zeros((M, 1), dtype='complex128')
        w_fir[int((M-1)/2)] = 1
        w_lut = np.ones((nspl, 1), dtype='complex128')
        w_full = np.concatenate((w_lut, w_fir), axis=0)
        print('Weights for Newton adaptation not given. Start from zero point')
    
    w_saved = np.copy(w_full)
    
    Nmse = []
    Mu = []
    flag = True   
    while flag:
        
        y, V, U = calc_out(x, d, w_full, nspl, pwr)
        e = d - y
        dw = solve(V, U, w_full[nspl:], e)
        
        w_temp = w_full + mu*dw
        y, V, U = calc_out(x, d, w_temp, nspl, pwr)
        e_new = d - y
        
        is_diverge = sl.nmse(d, e_new) > sl.nmse(d, e)

        while is_diverge:
            mu /= 2
            w_temp = w_full + mu*dw
            y, V, U = calc_out(x, d, w_temp, nspl, pwr)
            e_new = d - y
            is_diverge = sl.nmse(d, e_new) > sl.nmse(d, e)
            if mu < 1e-18:
                flag = False
                break
        
        mu *= 2
        if mu > 1:
            mu = 1
        w_full = w_temp
        
        # while loop stop condition:
        # (Difference between max elements of weights)
        # (If delta weights is too small, then we get the desired point)
        if np.max(np.abs(np.abs(w_full) - np.abs(w_saved))) < 1e-8:
            flag = False
        w_saved = w_full

        Nmse.append(sl.nmse(d, e_new))
        Mu.append(mu)

    y, V, U = calc_out(x, d, w_full, nspl, pwr)
    
    return  y, w_full, Nmse, Mu


def calc_out_hammerstein_cheby_1(x, d, w_full, nnl, pwr=1):
    """ 
        Hammerstein model with Cbeyshev polynomials nonlinearity for Newton-type adaptation
        x - input signal
        d - desired signal
        M - filter order (must be odd)
        nnl - number of weights for Cheby nonlinearity
        pwr - power for LUT output: output = lut(|x|) * (x**pwr)
    """
    M = w_full.size - nnl
    w_ch, w_fir = w_full[:nnl], w_full[nnl:]
    
    V = dpd.cheby_statement_vec(x, nnl, pwr)
    z = (V @ w_ch).reshape(-1,)
    U = dpd.fir_statement_vec(z, M, is_noncausal=True)
    y = (U @ w_fir).reshape(-1,)
    return y, V, U


def calc_out_hammerstein_lut_1(x, d, w_full, nspl, pwr=1):
    """ Hammerstein model with LUT nonlinearity for Newton-type adaptation
        x - input signal
        d - desired signal
        M - filter order (must be odd)
        nspl - spline number
        pwr - power for LUT output: output = lut(|x|) * (x**pwr)
    """

    M = w_full.shape[0] - nspl
    w_lut, w_fir = w_full[:nspl], w_full[nspl:]  
    
    V = dpd.lut_statement_vec(x, nspl, pwr=pwr)
    z = (V @ w_lut).reshape(-1,)
    U = dpd.fir_statement_vec(z, M, is_noncausal=True)
    y = (U @ w_fir).reshape(-1,)
    return y, V, U
    

def solve_hammerstein_lut_1(V, U, w_fir, e):
    """ Hammerstein solver with LUT nonlinearity for Newton-type adaptation
        V - lut statement vec
        U - fir statement vec
        w_fir - fir weights
        e - error
    """
    Vf = dpd.filter_lut_statement(V, w_fir, True)
    Ufull = np.concatenate((Vf, U), axis=1)
    H = np.conjugate(Ufull.T) @ Ufull
    grad = (np.conjugate(Ufull.T) @ e).reshape(-1,1)
    dw = pinv(H) @ grad
    return dw


# %% DCD solver

def complex_leading_DCD(R, b, M, Nu=8, Mb=32, a=1):
    """ DCD solver for system of complex linear equations
        M - one dimention of R
        Nu - number of iterations
        Mb - precision
        a - some initial value (1 recommended)
    """
    m = 0
    dw = np.zeros((M, 1), dtype='complex128')
    r = np.copy(b)
    for kk in range(Nu):
        nr = np.argmax(np.abs(np.real(r)))
        ni = np.argmax(np.abs(np.imag(r)))
        if np.abs(np.real(r[nr])) > np.abs(np.imag(r[ni])):
            s = 1
            n = nr
        else:
            s = 1j
            n = ni

        if s == 1:
            rtmp = np.real(r[n])
        else:
            rtmp = np.imag(r[n])
        while (np.abs(rtmp) <= (a/2)*R[n, n]):
            m = m + 1
            a = a / 2
            if m > Mb:
                return dw, r
        dw[n] = dw[n] + np.sign(rtmp)*s*a
        r = r - np.sign(rtmp)*s*a*R[:, n]
    return dw, r


# %% Parallel Hammerstein different adaptation approaches


def parallel_hammerstein_2_wrlsdcd(x, xa, d, M=13, lam=0.9995, mul=0.0001,
                                   Nu=16, Mb=64, T=np.array([0]),
                                   is_plot=False, is_write=True, is_retsig=False):
    N = len(x)
    if T.size != 1:
        K = T.shape[0]
    else:
        K = M
    sigma = np.ones((K,), dtype='complex128')*mul
    y = np.zeros((N,), dtype='complex128')
    e = np.zeros((N,), dtype='complex128')
    w = np.random.uniform(size=(K, 1)) + 1j*np.random.uniform(size=(K, 1))
    r = np.zeros((K,), dtype='complex128')
    R = np.diag(sigma)

    wa = np.random.uniform(size=(K, 1)) + 1j*np.random.uniform(size=(K, 1))
    ra = np.zeros((K,), dtype='complex128')
    Ra = np.diag(sigma)

    U = dpd.fir_statement_vec(x, M, T, is_noncausal=True)
    Ua = dpd.fir_statement_vec(xa, M, T, is_noncausal=True)

    for kk in range(N):
        u = np.reshape((U[kk, :]), (K, 1))
        R = lam*R + np.matmul(np.conj(u), np.transpose(u))
        ua = np.reshape((Ua[kk, :]), (K, 1))
        Ra = lam*Ra + np.matmul(np.conj(ua), np.transpose(ua))

        y[kk] = np.matmul(np.transpose(u), w) + np.matmul(np.transpose(ua), wa)
        e[kk] = d[kk] - y[kk]

        b = lam*r + e[kk]*np.conj(np.reshape(u, (K,)))
        dw, r = complex_leading_DCD(R, b, K, Nu, Mb)
        w = w + dw

        ba = lam*ra + e[kk]*np.conj(np.reshape(ua, (K,)))
        dwa, ra = complex_leading_DCD(Ra, ba, K, Nu, Mb)
        wa = wa + dwa

    plt.plot(np.abs(e))
    
    dpd_sig1 = np.matmul(U, w)
    dpd_sig2 = np.matmul(Ua, wa)
    dpd_sig = dpd_sig1 + dpd_sig2
    ws = np.vstack((w, wa))

    #dpd_sig = y

    dpd_sig.resize(N,)
    e_new = d-dpd_sig
    nmse = sl.nmse(d, e_new)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig, legend=('rx', 'synthesized', 'rx-synthesized'))
    nmse = sl.nmse(d, d-dpd_sig)
    if is_write:
        print('NMSE after finding solution:', nmse)

    info = ('''Model: Parallel Hammerstein, method: WRLS-DCD,
            FIR1''', M, 'taps, FIR2', M, 'taps')
    dop_info = ('lam', lam, 'mul', mul, 'Nu', Nu, 'Mb', Mb)

    if is_retsig:
        dd = dpd_sig
    else:
        dd = 0

    return e, ws, info, dop_info, dd



def parallel_hammerstein_2_Newton(x, d, M, xa=[0], sigma=0,
                              T1r=np.array([0]), T1i=np.array([0]),
                              T2r=np.array([0]), T2i=np.array([0]),
                              is_plot=False, is_write=True,
                              is_retsig=False, is_noncausal=True):
    """ Proposed model (Parallel Hammerstein 2 branches) Newton solution

        x - input signal
        d - desired signal
        M - number of filter weights
        xa - abs nonlinear part of signal
        sigma - can be useful in normalization of LS solution
        Txx - transform matrix for different cases:
            T1x - first filter (r - real part, i - imaginary part)
            T2x - second filter (for abs nonlinear part)
        is_plot - do we need to plot the results
        is_write - do we need to print the results
        is_retsig - do we need to return compensator signal
        is_noncausal - if True, then we eliminate filter delay

    """
    N = len(x)

    # If there is no abs, generate by ourselves
    if len(xa) == 1:
        xa = x * np.abs(x)

    # Generate matrix for LS solution (Contains all filter statements)
    U = dpd.fir_statement_vec(x, M, Tr=T1r, Ti=T1i, is_noncausal=is_noncausal)
    Ua = dpd.fir_statement_vec(xa, M, Tr=T2r, Ti=T2i, is_noncausal=is_noncausal)
    Us = np.hstack((U, Ua))
    del U, Ua

    step = 12000
    mu = 0.5
    wopt = np.zeros((M*2,1), dtype='complex128')
    for ii in range(0, N-step, step):

        U = Us[ii:ii+step, :]
        UH = np.transpose(np.conj(U))
        R = np.matmul(UH, U)
        e = d[ii:ii+step].reshape(step,1) - U @ wopt
        rxd = np.matmul(UH, e).reshape(M*2, 1)
        wopt = wopt + mu*pinv(R) @ rxd


    # Solving using LS and generating output signal [wsopt = R^(-1)*rd]
    wsopt = wopt
    dpd_sig = np.matmul(Us, wsopt).reshape(len(x),)

    # Calculating nmse and plotting
    nmse = sl.nmse(d, d-dpd_sig)
    info = ('''Model: Parallel Hammerstein, method: LS,
            FIR1''', M, 'taps, FIR2', M, 'taps')
    dop_info = (T1r, T1i, T2r, T2i)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig, legend=('desired', 'synthesized', 'error'))
    if is_write:
        print('NMSE after finding solution:', nmse)
    if is_retsig:
        None
    else:
        dpd_sig = 0

    return nmse, wsopt, info, dop_info, dpd_sig



def parallel_hammerstein_2_ls(x, d, M, xa=[0], n_branches=2, sigma=0,
                              T1r=np.array([0]), T1i=np.array([0]),
                              T2r=np.array([0]), T2i=np.array([0]),
                              is_plot=False, is_write=True,
                              is_retsig=False, is_noncausal=True):
    """ Proposed model (Parallel Hammerstein 2 branches) LS solution

        x - input signal
        d - desired signal
        M - number of filter weights
        xa - abs nonlinear part of signal
        sigma - can be useful in normalization of LS solution
        Txx - transform matrix for different cases:
            T1x - first filter (r - real part, i - imaginary part)
            T2x - second filter (for abs nonlinear part)
        is_plot - do we need to plot the results
        is_write - do we need to print the results
        is_retsig - do we need to return compensator signal
        is_noncausal - if True, then we eliminate filter delay

    """
    # If there is no abs, generate by ourselves
    if len(xa) == 1:
        xa = x * np.abs(x)

    # Generate matrix for LS solution (Contains all filter statements)
    U = dpd.fir_statement_vec(x, M, Tr=T1r, Ti=T1i, is_noncausal=is_noncausal)
    Ua = dpd.fir_statement_vec(xa, M, Tr=T2r, Ti=T2i, is_noncausal=is_noncausal)
    if n_branches == 2:
        Us = np.hstack((U, Ua)) 
    if n_branches == 1:
        Us = U
    del U, Ua

    # Solving using LS and generating output signal [wsopt = R^(-1)*rd]
    wsopt = dpd.get_ls(Us, d, sigma)
    dpd_sig = np.matmul(Us, wsopt)

    # Calculating nmse and plotting
    nmse = sl.nmse(d, d-dpd_sig)
    info = ('''Model: Parallel Hammerstein, method: LS,
            FIR1''', M, 'taps, FIR2', M, 'taps')
    dop_info = (T1r, T1i, T2r, T2i)

    if is_plot:
        pl.plot_psd(x, d, dpd_sig, d-dpd_sig, legend=('input','desired', 'synthesized', 'error'))
    if is_write:
        print('NMSE after finding solution:', nmse)
    if is_retsig:
        None
    else:
        dpd_sig = 0

    return nmse, wsopt, info, dop_info, dpd_sig


def parallel_hammerstein_n_ls(x, xa, d, M, is_plot=False, is_write=True, 
                              T=np.array([0]), is_retsig=False, is_noncausaly=False):
    """ Proposed model (Parallel Hammerstein n branches) LS solution """
    xa = x * xa
    xc = x * xa**2
    xb = x**2 * xa
    xd = x**2 * xa**2
    
    U = dpd.fir_statement_vec(x, M, T, is_noncausal=is_noncausaly)
    
    Ua = dpd.fir_statement_vec(xa, M, T, is_noncausal=is_noncausaly)
    Ub = dpd.fir_statement_vec(xb, M, T, is_noncausal=is_noncausaly)
    Uc = dpd.fir_statement_vec(xc, M, T, is_noncausal=is_noncausaly)
    Ud = dpd.fir_statement_vec(xd, M, T, is_noncausal=is_noncausaly)

    Us = np.hstack((U, Ua, Ub, Uc, Ud))
    Us = U

    wsopt = dpd.get_ls(Us, d)
    dpd_sig = np.matmul(Us, wsopt)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig)
    nmse = sl.nmse(d, d-dpd_sig)
    if is_write:
        print('NMSE after finding solution:', nmse)
    info = ('''Model: Parallel Hammerstein, method: LS,
            FIR1''', M, 'taps, FIR2', M, 'taps')
    dop_info = None
    if is_retsig:
        None
    else:
        dpd_sig = 0
    return nmse, wsopt, info, dop_info, dpd_sig


def parallel_hammerstein_2_wrls(txsn, d, M=13, lam=0.999, mul=0.0001, 
                                T=np.array([0]), is_plot=False, is_write=True):
    N = len(txsn)
    llam = (1/lam)

    if T.size != 1:
        K = T.shape[0]
    else:
        K = M

    w = np.zeros((K, 1), dtype='complex128')
    wa = np.zeros((K, 1), dtype='complex128')
    ws = np.vstack((w, wa))

    sigma = np.ones((K*2,), dtype='complex128')*mul
    P = np.diag(sigma)
    y = np.zeros((N,), dtype='complex128')
    e = np.zeros((N,), dtype='complex128')

    U = dpd.fir_statement_vec(txsn, M, T, is_noncausal=True)
    Ua = dpd.fir_statement_vec(txsn*np.abs(txsn), M, T, is_noncausal=True)
    Us = np.hstack((U, Ua))
    k = np.zeros((1, 1), dtype='complex128')

    for nn in range(len(d)):
        u = (Us[nn, :])
        u.resize(2*K, 1)
        y[nn] = np.matmul(np.transpose(u), ws)
        e[nn] = d[nn] - y[nn]
        k = np.matmul(P, np.conj(u)) / (lam + np.matmul(np.matmul(np.transpose(u), P), np.conj(u)))
        ws = ws + k*e[nn]
        P = llam*(P - np.matmul(np.matmul(k, np.transpose(u)), P))

    dpd_sig = np.matmul(Us, ws)
    dpd_sig.resize(N,)
    e_new = d-dpd_sig

    nmse = sl.nmse(d, e_new)

    if is_plot:
        pl.plot_psd(d, dpd_sig, d-dpd_sig)
    nmse = sl.nmse(d, d-dpd_sig)
    if is_write:
        print('NMSE after finding solution:', nmse)

    info = ('''Model: Parallel Hammerstein, method: WRLS,
            FIR1''', M, 'taps, FIR2', M, 'taps')
    dop_info = ('lam', lam, 'mul', mul)

    return nmse, ws, info, dop_info, dpd_sig

