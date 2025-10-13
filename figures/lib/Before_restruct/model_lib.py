# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:41:28 2020

@author:
    Bakhurin Sergey (b00741681)
    Sayfullin Karim (swx959511)
    
@description:
    This library provides models to use in dpd structures

"""

import numpy as np


def lut_1d(x, w, nspl):
    ''' '''
    aa = np.abs(x)*nspl
    addra = np.floor(aa)
    da = aa - addra
    y = w[addra] + (w[addra+1] - w[addra])*da
    return y, addra


def parallel_lms(x, y, lut_info, nspl, nlin):
    """ Implementing lms adaptation of sl 1d lut parallel model """
    # if (np.sum(lut_info[:, 0]) != 0) or (lut_info[:, 1] != 6):
    #     print('Error! Check lut_info')

    N = len(x)
    NLUT = np.sum(lut_info[:, 1])
    MU = np.ones((NLUT+1), dtype='complex64')
    max_delay = np.max(np.abs(lut_info))*2
    ncoeff = nspl * NLUT
    V = np.zeros((ncoeff, 1), dtype='complex64')

    d = y-x
    a = np.abs(x)

    for nn in range(max_delay, N-max_delay):
        None


def parallel_ls(xa, xb, lut_info, nspl, nlin):
    """ Return statement vector for  parallel LUT&FIR dpd model """
    ncoeff = 0

    for ii in range(lut_info.shape[1]):
        if lut_info[0, ii] == 0:  # 1D LUT
            ncoeff = ncoeff + nspl
        elif lut_info[0, ii] == 1:  # 2D LUT
            ncoeff = ncoeff + nspl ** 2

    ncoeff = ncoeff + nlin
    max_delay = np.max(np.abs(lut_info))

    aa = abs(xa) * nspl
    addra = np.floor(aa)
    dxa = aa - addra

    ab = abs(xb) * nspl
    addrb = np.floor(ab)
    dxb = ab - addrb  # For 2D

    N = len(xa)
    NLUT = lut_info.shape[1]
    U = np.zeros((N, ncoeff), dtype='complex64')

    for ii in range(max_delay, N-max_delay):
        index = 0
        for ll in range(NLUT):
            if lut_info[0, ll] == 0:  # 1D LUT

                if lut_info[1, ll] == 0:  # LUT
                    ds = lut_info[3, ll]
                    pos = index + int(addra[ii-ds])
                    U[ii, pos] = (1 - dxa[ii-ds])
                    U[ii, pos+1] = dxa[ii-ds]

                if lut_info[1, ll] == 1:  # SL LUT
                    dl = lut_info[2, ll]
                    ds = lut_info[3, ll]
                    mul = xa[ii - dl]
                    pos = index + int(addra[ii-ds])
                    U[ii, pos] = (1 - dxa[ii-ds]) * mul
                    U[ii, pos+1] = dxa[ii-ds] * mul

                if lut_info[1, ll] == 2:  # SIL LUT
                    dl = lut_info[2, ll]
                    ds = lut_info[3, ll]
                    du = lut_info[4, ll]
                    dv = lut_info[5, ll]
                    mul = xa[ii-dl] * xa[ii-du] * np.conjugate(xa[ii-dv])
                    pos = index + int(addra[ii-ds])
                    U[ii, pos] = (1 - dxa[ii-ds]) * mul
                    U[ii, pos+1] = dxa[ii-ds] * mul

                if lut_info[1, ll] == 3:  # ABS LUT
                    dl = lut_info[2, ll]
                    ds = lut_info[3, ll]
                    du = lut_info[4, ll]
                    mul = xa[ii-dl] * np.abs(xa[ii-du])
                    pos = index + int(addra[ii-ds])
                    U[ii, pos] = (1 - dxa[ii-ds]) * mul
                    U[ii, pos+1] = dxa[ii-ds] * mul

                index = index + nspl

            if lut_info[0, ll] == 1:  # 2D LUT
                ds = lut_info[3, ll]
                pos = index + int(addrb[ii-ds])*nspl + int(addra[ii-ds])
                a00 = pos
                a01 = pos+1
                a10 = a00 + nspl
                a11 = a01 + nspl

                if lut_info[1, ll] == 0:  # simple 2D LUT
                    mul = 1

                elif lut_info[1, ll] == 1:  # SL 2D LUT
                    dl = lut_info[2, ll]
                    mul = xa[ii - dl]

                elif lut_info[1, ll] == 2:  # SIL 2D LUT
                    dl = lut_info[2, ll]
                    du = lut_info[4, ll]
                    dv = lut_info[5, ll]
                    mul = xa[ii - dl] * xa[ii - du] * np.conjugate(xa[ii - dv])

                elif lut_info[1, ll] == 3:  # SIL 2D v2
                    dl = lut_info[2, ll]
                    du = lut_info[4, ll]
                    dv = lut_info[5, ll]
                    mul = xa[ii - dl] * xb[ii - du] * np.conjugate(xb[ii - dv])

                elif lut_info[1, ll] == 4:  # SIL 2D ABS
                    dl = lut_info[2, ll]
                    du = lut_info[4, ll]
                    mul = xa[ii - dl] * np.abs(xa[ii - du])

                elif lut_info[1, ll] == 5:  # SIL 2D ABS v2
                    dl = lut_info[2, ll]
                    du = lut_info[4, ll]
                    mul = xa[ii - dl] * np.abs(xb[ii - du])

                U[ii, a00] = (1 - dxa[ii - ds]) * (1 - dxb[ii - ds]) * mul
                U[ii, a01] = (    dxa[ii - ds]) * (1 - dxb[ii - ds]) * mul
                U[ii, a10] = (1 - dxa[ii - ds]) * (dxb[ii - ds]    ) * mul
                U[ii, a11] = (    dxa[ii - ds]) * (dxb[ii - ds]    ) * mul

                index = index + nspl * nspl

        dlin = int((nlin-1) / 2)
        for m in range(nlin):
            U[ii, index+m] = xa[ii - m - dlin]

    return U
