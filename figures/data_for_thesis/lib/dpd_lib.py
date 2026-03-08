# -*- coding: utf-8 -*-

"""
Created on

@author:
    Sayfullin Karim (swx959511)

@description:
    This library provides tools for solving LS, and some additional useful func

"""

import numpy as np
import scipy.linalg

# %% Mixers and so on...


def cheby_poly(x, ord):
    """ Chebyshev polynomials of 1 kind """
    #ord = ord + 1
    out = np.zeros((ord,), dtype='complex128')
    out[0] = 1
    out[1] = x
    for ii in range(2, ord):
        out[ii] = 2*x*out[ii-1] - out[ii-2]
            
    return out#[np.array([1,2,3])]


def mixer(x, f, fs=1):
    """ Implementing digital mixer block """
    N = len(x)
    t = np.linspace(0, (N-1), N) - N/2.0
    return x * np.exp(1j * 2 * np.pi * f/fs * t)


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
    return np.matmul(scipy.linalg.pinv(R), rd)


def filter_lut_statement(V, w, is_noncausal):
    """ Filtering LUT statement vector over time """

    Vf = np.copy(V) * 0.0
    Vf2 = np.copy(V) * 0.0
    for ii in range(V.shape[1]):
        Vf[:, ii] = fir_filter(V[:, ii], w, is_noncausal=is_noncausal).reshape(-1,)
        #Vf[:, ii] = np.convolve(V[:, ii], w.reshape((-1,)), mode='same').reshape(-1,)

    return Vf


def cheby_statement_vec(x, ord, pwr):
    """ Implementation of statement vector for Cheby polynomials """
    N = len(x)
    V = np.zeros((N, ord), dtype='complex128')
    
    for ii in range(N):
        V[ii, :] = cheby_poly(np.abs(x[ii]), ord)*x[ii]**pwr
    
    return V


def lut_statement_vec(x, nspl, pwr=1):
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


# %% Filtered LS solution (This section is not working)

def get_cnst_filter_matrix(h, N):
    """ """
    F = np.zeros((N, N), dtype='complex64')
    M = len(h)
    for ii in range(N-M+1):
        F[ii:ii+M, ii] = h
    jj = 0
    for ii in range(N-M+1, N):
        F[ii:, ii] = h[0:(M-jj)-1]
        jj = jj+1
    return (F)


def get_ls_filtered(U, F, d):
    """ Return optimal weights, calculated using filtered LS """
    FU = np.matmul(F, U)
    FUH = np.transpose(np.conjugate(FU))
    R_inv = np.linalg.pinv(np.matmul(FUH, FU))
    ry = np.matmul(FUH, d)
    return np.matmul(R_inv, ry)
