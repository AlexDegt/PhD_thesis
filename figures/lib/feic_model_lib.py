# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:11:48 2021

@author: sWX959511
"""

import numpy as np
import scipy.signal as signal
from scipy.linalg import pinv as pinv
import dpd_Sergey_lib as dpd

#import support_lib as sl


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


def Hammerstein_out(x, w_lut, w_fir, pwr):
    
    N = len(x)
    y = np.zeros((N,), dtype='complex128')

    xa = np.abs(x)*w_lut.shape[0]
    addr = (np.floor(xa)).astype('int')
    dx = xa - addr

    for ii in range(N):
        m = w_lut[addr[ii]] + dx[ii]*(w_lut[addr[ii]+1] - w_lut[addr[ii]])
        y[ii] = m*(x[ii])**pwr
    
    return fir_filter(y, w_fir, is_noncausal=False)

def Hammerstein_out_2dlut(x, xb, w_lut, w_fir, pwr, lut_ind=None, hist=[]):
    
    N = len(x)
    y = np.zeros((N,), dtype='complex128')
    nspl_sqr = np.size(w_lut)
    nspl = np.sqrt(nspl_sqr).astype('int')

    model = np.zeros((6, 1), dtype = 'int')
    model[0, 0] = 6
    model[1, 0] = 2   
    V = dpd.vand2d(x, xb, model, nspln = nspl, mlin = 0, bits_interp = -1)
    
    if np.size(lut_ind) <= 1:
        for ii in range(N):
            y[ii] = V[ii, :] @ w_lut
    else:
        for ii in range(N):
            y[ii] = V[ii, lut_ind] @ w_lut[lut_ind]
                
    if hist == []:
        return fir_filter(y, w_fir, is_noncausal=False)
    else:
        y = np.hstack([np.flip(hist.T), y.reshape((1, N))]).reshape((-1))
        return np.convolve(y, w_fir, mode='valid')

    
def lut_statement_matrix(x, nspl, pwr=1):
    """ Implementation of statement vector for LUT """
    N = len(x)
    xa = np.abs(x)*nspl
    addr = (np.floor(xa)).astype('int')
    dx = xa - addr
    V = np.zeros((N, nspl), dtype='complex128')

    for ii in range(N):
        V[ii, addr[ii]] = (1 - dx[ii])*x[ii]**pwr
        V[ii, addr[ii]+1] = dx[ii]*x[ii]**pwr

    return V


def get_ls(U, d, sigma=0):
    """ Return optimal weights, calculated using LS """
    # LS
    # wopt = R^(-1)*r = (UHU)UHd
    UH = np.transpose(np.conjugate(U))
    R = np.matmul(UH, U)
    if sigma == 0:
        None
    else:
        R = R + np.eye(R.shape[0])*sigma
    rd = np.matmul(UH, d)
    return np.matmul(pinv(R), rd)


def fir_statement_vec(x, M, Tr=np.array([0]), Ti=np.array([0]), is_noncausal=False):
    """ Implementation of statement vector for FIR 
        x - input signal. Shape: (N,)
        M - fir taps num
    """
    if len(x.shape) > 1:
        x = x.reshape(-1,)
    n = x.size
    m = M - 1
    k = int(m/2)
    c = int((M+1)/2)-1
    U = np.zeros([n, m+1], dtype='complex128')

    if is_noncausal:
        U[:, c] = x
        # Moving to the left:
        for mm in range(k):
            U[0:n-k+mm, mm] = x[k-mm:]
        # Moving to the right:
        for mm in range(c, M):
            U[mm-c:, mm] = x[0:n-(mm-c)]
    else:
        U[:, 0] = x
        for mm in range(1, m+1):
            U[mm:, mm] = x[0:-mm]

    if len(Tr) != 1 and len(Ti) == 1:
        Unew = np.zeros([n, Tr.shape[0]], dtype='complex128')
        for ii in range(n):
            Unew[ii, :] = np.matmul(Tr, U[ii, :])
        return Unew

    elif len(Tr) != 1 and len(Ti) != 1:
        Unewr = np.zeros([n, Tr.shape[0]], dtype='float64')
        Unewi = np.zeros([n, Ti.shape[0]], dtype='float64')
        for ii in range(n):
            Unewr[ii, :] = np.matmul(Tr, np.real(U[ii, :]))
            Unewi[ii, :] = np.matmul(Ti, np.imag(U[ii, :]))
        Unew = Unewr + 1j*Unewi
        return Unew

    else:
        return U


def fir_filter(x, w, is_noncausal=False):
    """ Perform filtering of the signal """
    if x.shape[0] == w.shape[0]:
        return np.matmul(x, w)
    else:
        U = fir_statement_vec(x, len(w), is_noncausal=is_noncausal)
        return U @ (w)


def filter_lut_statement(V, w, is_noncausal):
    """ Filtering LUT statement vector over time """

    Vf = np.copy(V) * 0.0
    Vf2 = np.copy(V) * 0.0
    for ii in range(V.shape[1]):
        Vf[:, ii] = fir_filter(V[:, ii], w, is_noncausal=is_noncausal).reshape(-1,)
        #Vf[:, ii] = np.convolve(V[:, ii], w.reshape((-1,)), mode='same').reshape(-1,)

    return Vf