# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:27:53 2021

@author: dWX1065688
"""

import plot_lib as pl
import scipy.signal as signal
import numpy as np
import support_lib as sl
import feic_model_lib as feic_model

def backprop_dcd_lms(coef, bp_input, data, model_data, bp_delays, adapt_param):
    b2, R2, u2, bp_buf2, e_lut2 = bp_input
    x, d = data
    addr, dx, lam, lut_upd_len, M, nspl, pwr = model_data
    Nu, Mb, mu_dcd, mu_lms = adapt_param
    dcd_block_fir, bp_wait, lut_upd_len  = bp_delays
    w_lut2, w_fir2 = coef
    delay_fir = int((M-1)/2)
    N = len(x)
    curr_state = 'fir_acc'
    cnt = 0
    for ii in range(0, N):
        l2 = w_lut2[addr[ii]] + dx[ii]*(w_lut2[addr[ii]+1] - w_lut2[addr[ii]])
        f2 = l2 * (x[ii])**pwr
        u2 = np.vstack((f2, u2[:-1])) 
        y2 = np.matmul(np.transpose(w_fir2), u2)  
        e0 = d[ii] - y2
        bp_buf2 = np.vstack((e0, bp_buf2[:-1]))
        e_lut2[ii] = np.matmul(np.transpose(np.conj(np.flip(w_fir2))), bp_buf2) 
        if ii > (delay_fir*2 + 0):
            if curr_state == 'fir_acc':
                cnt += 1
                R2 = lam*R2 + np.matmul(np.conj(u2), np.transpose(u2))
                b2 = lam*b2 + (e0*np.conj(u2)).reshape((M,))           
                if cnt >= dcd_block_fir:
                    dw2, b2 = feic_model.complex_leading_DCD(R2, b2, M, Nu, Mb)
                    w_fir2 = w_fir2 + mu_dcd*dw2
                    curr_state = 'bp_wait'
                    cnt = 0            
            if curr_state == 'bp_wait':
                cnt += 1
                if cnt >= bp_wait:
                    curr_state = 'lut_upd'
                    cnt = 0      
            elif curr_state == 'lut_upd':
                cnt += 1
                nn = addr[ii-delay_fir*2]
                ''' Simple LMS '''
                dw_lut1 = (1-dx[ii-delay_fir*2])*e_lut2[ii]*np.conj(x[ii-delay_fir*2]**pwr)#*(1/np.abs(x[ii-delay_fir*2])**2)
                dw_lut2 = (dx[ii-delay_fir*2])*e_lut2[ii]*np.conj(x[ii-delay_fir*2]**pwr)#*(1/np.abs(x[ii-delay_fir*2])**2)
                w_lut2[nn] = w_lut2[nn] + mu_lms*dw_lut1
                w_lut2[nn+1] = w_lut2[nn+1] + mu_lms*dw_lut2            
                if cnt >= lut_upd_len:
                    curr_state = 'fir_acc'
                    cnt = 0
    return w_lut2, w_fir2

def damped_newton_solver(x, d, w_lut, w_fir, mu, sym_iter_num, shift, pwr = 2):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr)
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old, z = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e_old = d - y_old      
        dw_lut, dw_fir = get_delta_newton(w_lut, w_fir, V, z, e_old, M, nspl)
        w_lut_tmp = w_lut + mu*dw_lut
#        w_fir_tmp = w_fir
        w_fir_tmp = w_fir + mu*dw_fir
        y_new, _ = sl.hammerstein_forward(M, w_lut_tmp, w_fir_tmp, V)
        e_new = d - y_new
        nmse_old = sl.nmse(d[shift:-shift], e_old[shift:-shift])
        error = nmse_old - sl.nmse(d[shift:-shift], e_new[shift:-shift])
        if (error >= 0):
            mu *= 2
            if mu > 1:
                mu = 1
            w_lut, w_fir = w_lut_tmp, w_fir_tmp
            e = e_new
        else:
            flag = True
            while flag:
                mu /= 1.5
                w_lut_tmp_ = w_lut + mu*dw_lut
#                w_fir_tmp_ = w_fir
                w_fir_tmp_ = w_fir + mu*dw_fir
                y_diverge, _ = sl.hammerstein_forward(M, w_lut_tmp_, w_fir_tmp_, V)
                e_diverge = d - y_diverge
                error = nmse_old - sl.nmse(d[shift:-shift], e_diverge[shift:-shift])
                if error >= 0:
                    flag = False
                    w_lut, w_fir = w_lut_tmp_, w_fir_tmp_
                    e = e_diverge
                print(mu, i, 'diverges')
        dataset_part_nmse[i] = sl.nmse(d[shift:-shift], e[shift:-shift])
        dataset_part_mse[i] = sl.mse(e[shift:-shift])
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, x, d, dataset_part_nmse, dataset_part_mse, e

def ALS_solver(x, d, w_lut, w_fir, mu=0.5, epoch_size=10, ls_epoch_size=1, trans_size=0, pwr=1):
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr=pwr)
    for epoch in range(epoch_size):
        ''' FIR adatation part '''
        z = V @ w_lut
        U = sl.fir_matrix_generate(z, M)
        print('FIR adaptation epoch', epoch)
        w_fir = LS_solver(U, d, w_fir, mu=mu, epoch_size=ls_epoch_size, trans_size=trans_size)
        ''' LUT adatation part '''
        V_f = sl.fir_filtering_matrix_conv(V, w_fir)
        print('LUT adaptation epoch', epoch)
        w_lut = LS_solver(V_f, d, w_lut, mu=mu, epoch_size=ls_epoch_size, trans_size=trans_size)
    return w_lut, w_fir

def LS_solver(S, d, w, nf=[], mu=0.5, epoch_size=1, trans_size=0):
    S_H = np.conj(S.T)
    y = S @ w
    e = d - y
    for epoch in range(epoch_size):
        dw = np.linalg.pinv(S_H @ S) @ S_H @ e
        w = w + mu*dw
        y = S @ w
        e = d - y
        if nf == []:
            if trans_size > 0:
                NMSE = sl.nmse(d[trans_size:-trans_size], e[trans_size:-trans_size])
            else:
                NMSE = sl.nmse(d, e)
        else:
            if trans_size > 0:
                NMSE = sl.nmse_nf(d[trans_size:-trans_size], \
                                   e[trans_size:-trans_size], \
                                   nf[trans_size:-trans_size])
            else:
                NMSE = sl.nmse_nf(d, e, nf)
        print('LS epoch =', epoch, ', NMSE =', NMSE)
    return w

def newton_solver(x, d, nf, w_lut, w_fir, mu, sym_iter_num, shift, pwr=2):
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr=pwr)
    dataset_part_nmse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old, z = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e_old = d - y_old      
        dw_lut, dw_fir = get_delta_newton(w_lut, w_fir, V, z, e_old, M, nspl)
        w_lut = w_lut + mu*dw_lut
        w_fir = w_fir + mu*dw_fir
        y_new, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e = d - y_new
        if shift > 0:
            dataset_part_nmse[i] = sl.nmse_nf(d[shift:-shift], e[shift:-shift], nf[shift:-shift])
        else:
            dataset_part_nmse[i] = sl.nmse_nf(d, e, nf)
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, dataset_part_nmse, z, e

def newton_solver_rs(x, d, nf, w_lut, w_fir, mu, sym_iter_num, shift, pwr=2):
    d_size = np.size(d)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr=pwr)
    V = signal.resample(V, d_size)
    dataset_part_nmse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old, z = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e_old = d - y_old      
        dw_lut, dw_fir = get_delta_newton(w_lut, w_fir, V, z, e_old, M, nspl)
        w_lut = w_lut + mu*dw_lut
        w_fir = w_fir + mu*dw_fir
        y_new, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e = d - y_new
        if shift > 0:
            dataset_part_nmse[i] = sl.nmse_nf(d[shift:-shift], e[shift:-shift], nf[shift:-shift])
        else:
            dataset_part_nmse[i] = sl.nmse_nf(d, e, nf)
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, dataset_part_nmse, z, e

def GD_solver_rs(x, d, w_lut, w_fir, mu, sym_iter_num, shift, pwr=2):
    '''
        Batch Gradient Descent solver
    '''
    d_size = np.size(d)
    x_size = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    dw_lut_prev = np.zeros((nspl,), dtype='complex128')
    dw_fir_prev = np.zeros((M,), dtype='complex128')
    moment = 0.9
    V = sl.lut_matrix_generate(x, nspl, pwr=pwr)
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        z = V @ w_lut
        z = signal.resample(z, d_size)
        U = sl.fir_matrix_generate(z, M)
        y = U @ w_fir
        e = d - y
        U_bp = sl.fir_matrix_generate(e, M)
        e_bp = U_bp @ np.conj(w_fir[::-1])
        e_bp = signal.resample(e_bp, x_size)
        dw_fir = moment*dw_fir_prev-np.conj(U).T @ e
        dw_lut = moment*dw_lut_prev-np.conj(V).T @ e_bp
        dw_fir_prev = dw_fir
        dw_lut_prev = dw_lut
        w_lut = w_lut - mu*dw_lut
        w_fir = w_fir - mu*dw_fir
        z = V @ w_lut
        z = signal.resample(z, d_size)
        U = sl.fir_matrix_generate(z, M)
        y = U @ w_fir
        e = d - y
        if shift > 0:
            dataset_part_nmse[i] = sl.nmse(d[shift:-shift], e[shift:-shift])
            dataset_part_mse[i] = sl.mse(e[shift:-shift])
        else:
            dataset_part_nmse[i] = sl.nmse(d, e)
            dataset_part_mse[i] = sl.mse(e)
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    return w_lut, w_fir, y, dataset_part_nmse, dataset_part_mse, e

def SGD_solver_rs(x, d, w_lut, w_fir, mu, sym_iter_num, shift, rs, w_fir_rs, pwr=2):
    '''
        Stochastic Gradient Descent solver
    
        FUNCTION NEEDS DEBUG!!!
    
        Example of w_fir_rs filter:
        L = 501
        cutoff = 0.1205
        trans_width = 0.005
        h = signal.remez(L, [0, cutoff, cutoff + trans_width, 0.5], [1, 0])
    '''
    x_size = np.size(x)
    d_size = np.size(d)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    M_rs = np.size(w_fir_rs)
    V = sl.lut_matrix_generate(x, nspl, pwr=pwr)
    fir_buffer = np.zeros((M,), dtype='complex128')
    fir_rs_buffer = np.zeros((M_rs,), dtype='complex128')
    err_buffer = np.zeros((M,), dtype='complex128')
    err_rs_buffer = np.zeros((M_rs,), dtype='complex128')
    y = np.zeros((d_size,), dtype='complex128')
    e = np.zeros((d_size,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    rs_ratio = int(x_size//rs)
    model_state = 'fir_wait'
    cnt = 0
    fir_wait = 1
    lut_wait = 1
    bp_wait = int(M_rs//2)+int(M//2)*rs_ratio
    V_q = np.zeros((bp_wait, nspl), dtype='complex128')
    for epoch in range(sym_iter_num):
    # for epoch in range(1):
        for x_sample in range(x_size-1):
            # Forward
            V_q = np.vstack([V[x_sample, :], V_q[:-1, :]])
            z = V[x_sample, :] @ w_lut
            fir_rs_buffer = np.hstack([z, fir_rs_buffer[:-1]])
            z_filt = fir_rs_buffer @ w_fir_rs
            if x_sample % rs_ratio == 0:
                d_sample = int(x_sample//rs_ratio)
                fir_buffer = np.hstack([z_filt, fir_buffer[:-1]])
                out = y[d_sample] = fir_buffer @ w_fir
                e_out = e[d_sample] = d[d_sample] - out
                # Backward
                err_buffer = np.hstack([e_out, err_buffer[:-1]])
                e_bp = err_buffer @ np.conj(w_fir[::-1])
                err_rs_buffer = np.hstack([e_bp, err_rs_buffer[:-1]])
                e_lut = err_rs_buffer @ np.conj(w_fir_rs[::-1])
                if model_state == 'fir_wait':
                    dw_fir = -mu*np.conj(fir_buffer)*e_out
                    cnt += 1
                    if cnt >= fir_wait:
                        model_state = 'bp_wait'
                        cnt = 0
                if model_state == 'bp_wait':
                    cnt += 1
                    if cnt >= bp_wait:
                        model_state = 'lut_wait'
                        cnt = 0  
                if model_state == 'lut_wait':
                    dw_lut = -mu*np.conj(V_q[-1, :])*e_lut
                    cnt += 1
                    if cnt >= lut_wait:
                        model_state = 'upd_coef'
                        cnt = 0
                if model_state == 'upd_coef':
                # Coefficients update
                    w_lut = w_lut - dw_lut
                    w_fir = w_fir - dw_fir
                    model_state = 'fir_wait'
            else:
                e_bp = 0
                err_rs_buffer = np.hstack([e_bp, err_rs_buffer[:-1]])
        if shift > 0:
            dataset_part_nmse[epoch] = sl.nmse(d[shift:-shift], e[shift:-shift])
            dataset_part_mse[epoch] = sl.mse(e[shift:-shift])
        else:
            dataset_part_nmse[epoch] = sl.nmse(d, e)
            dataset_part_mse[epoch] = sl.mse(e)
        print(epoch, mu, 'NMSE:', dataset_part_nmse[epoch])
    return w_lut, w_fir, y, dataset_part_nmse, dataset_part_mse, e

def gradient_descent_solver(x, d, w_lut, w_fir, mu, sym_iter_num, shift, pwr = 2):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr)
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y, z = sl.hammerstein_forward(M, w_lut, w_fir, V)
        # d = d*signal.tukey(np.size(d), 0.002)
        # y = y*signal.tukey(np.size(y), 0.002)
        # e = d - y      
        # e = e*signal.tukey(np.size(e), 0.002)
        
        V_f = sl.fir_filtering_matrix_conv(V, w_fir)
        U = sl.fir_matrix_generate(z, M)
        V_full = np.hstack((V_f, U))
        V_full_H = np.conj(V_full).T
        dw_full = V_full_H @ e
        dw_lut, dw_fir = dw_full[0:nspl], dw_full[nspl:nspl + M]  

        w_lut = w_lut + mu*dw_lut
        w_fir = w_fir + mu*dw_fir
        y, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e = d - y
        dataset_part_nmse[i] = sl.nmse(d[shift:-shift], e[shift:-shift])
        dataset_part_mse[i] = sl.mse(e[shift:-shift])
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, x, d, dataset_part_nmse, dataset_part_mse, e

def DCD_solver(x, d, w_lut, w_fir, mu, sym_iter_num, pwr = 2, Nu = 32, Mb = 64, RT_ratio=0.0001):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr)
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    R = 0.001*(np.eye(M+nspl)+1j*np.eye(M+nspl))
    b = np.zeros((M+nspl), dtype='complex128')
    R_arr = np.zeros((M+nspl, M+nspl, sym_iter_num), dtype='complex128')
    for i in range(sym_iter_num):
        y, z = sl.hammerstein_forward(M, w_lut, w_fir, V)        
        
        V_f = sl.fir_filtering_matrix_conv(V, w_fir)
        U = sl.fir_matrix_generate(z, M)
        V_full = np.hstack((V_f, U))
        V_full_H = np.conj(V_full).T
        R = V_full_H @ V_full
        b = V_full_H @ d
        Rmax0 = np.max(np.abs(R[0:nspl, 0:nspl]))
        R[0:nspl, 0:nspl] /= Rmax0
        b[0:nspl] /= Rmax0
        R1 = R.copy()
        R1[0:nspl, 0:nspl] = 0
        Rmax1 = np.max(np.abs(R1))
        R[nspl:nspl+M, 0:nspl] /= Rmax1
        R[0:nspl+M, nspl:nspl+M] /= Rmax1
        b[nspl:nspl+M] /= Rmax1
        R += RT_ratio*(np.eye(M+nspl)+1j*np.eye(M+nspl))
        R_arr[:, :, i] = R
        dw_full, b = feic_model.complex_leading_DCD(R, b, M+nspl, Nu, Mb)
        dw_full = dw_full.reshape((-1))
        dw_lut, dw_fir = dw_full[0:nspl], dw_full[nspl:nspl + M]

        w_lut = w_lut + mu*dw_lut
        w_fir = w_fir + mu*dw_fir
        y, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
        e = d - y
        dataset_part_nmse[i] = sl.nmse(d[24:-24], e[24:-24])
        dataset_part_mse[i] = sl.mse(e[24:-24])
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, x, d, dataset_part_nmse, dataset_part_mse, e, R_arr

def get_delta_newton_gain(w_lut, w_fir, V, z, Vg, gain, curr_sig, e, M, nspl, stype_num):
    V_f = gain[curr_sig]*sl.fir_filtering_matrix_conv(V, w_fir)
    U = gain[curr_sig]*sl.fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U, Vg))
    V_full_H = np.conj(V_full).T
    dw_full = np.linalg.pinv(V_full_H @ V_full, rcond=1e-6) @ V_full_H @ e
    dw_lut, dw_fir, dgain = dw_full[0:nspl], dw_full[nspl:nspl + M], dw_full[nspl + M:nspl + M + stype_num]
    return dw_lut, dw_fir, dgain

def get_delta_newton(w_lut, w_fir, V, z, e, M, nspl):
    V_f = sl.fir_filtering_matrix_conv(V, w_fir)
    U = sl.fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U))
    V_full_H = np.conj(V_full).T
    dw_full = np.linalg.pinv(V_full_H @ V_full, rcond=1e-8) @ V_full_H @ e
    dw_lut, dw_fir = dw_full[0:nspl], dw_full[nspl:nspl + M]
    return dw_lut, dw_fir

def update_coef_newton(w_lut, w_fir, V, z, e, M, nspl, mu):
    V_f = sl.fir_filtering_matrix_conv(V, w_fir)
    U = sl.fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U))
    V_full_H = np.conj(V_full).T
    dw_full = np.linalg.pinv(V_full_H @ V_full) @ V_full_H @ e
    w_full = np.vstack((w_lut.reshape((nspl,1)), w_fir.reshape((M,1)))).reshape((-1,))
    w_full = w_full + mu*dw_full
    w_lut, w_fir = w_full[0:nspl], w_full[nspl:nspl + M]
    return w_lut, w_fir

def damped_newton_fir_adapt(x, d, w, mu, sym_iter_num):
    N = np.size(x)
    M = np.size(w)
    e = np.zeros((N,), dtype='complex128')
    U = sl.fir_matrix_generate(x, M)
    U_H = np.conj(U.T)
    dataset_part_nmse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old = U @ w
        e_old = d - y_old
        dw = np.linalg.pinv(U_H @ U) @ U_H @ e_old
        w_tmp = w + mu*dw
        y_new = U @ w_tmp
        e_new = d - y_new
        nmse_old = sl.nmse(x[24:-24], e_old[24:-24])
        error = nmse_old - sl.nmse(x[24:-24], e_new[24:-24])
        if (error >= 0):
            mu *= 2
            if mu > 1:
                mu = 1
            w = w_tmp
            e = e_new
        else:
            flag = True
            while flag:
                mu /= 1.5
                w_tmp_ = w + mu*dw
                y_diverge = U @ w_tmp_
                e_diverge = d - y_diverge
                error = nmse_old - sl.nmse(x[24:-24], e_diverge[24:-24])
                if error >= 0:
                    flag = False
                    w = w_tmp_
                    e = e_diverge
                print(mu, i, 'diverges')
        dataset_part_nmse[i] = sl.nmse(d[24:-24], e[24:-24])
        print(mu, dataset_part_nmse[i], i)
    generated = U @ w
    return w, generated, x, d, dataset_part_nmse, e

def lms_solver(x, d, w_lut, w_fir, mu_start, sym_iter_num):
    mu = mu_start
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sl.lut_matrix_generate(x, nspl, pwr = 2)
    y = np.zeros((N,), dtype='complex128')
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    for k in range(sym_iter_num):
#        if (k > int(sym_iter_num/2) - 1):
#            mu /= 10
        y, z = sl.hammerstein_forward(M, w_lut, w_fir, V)
        y = y.reshape((-1,))
        e = d - y
        if np.var(e) > 1e+10:
            print('The algorithm diverges, try decreasing the step')
            break
        V_f = sl.fir_filtering_matrix_conv(V, w_fir)
        U = sl.fir_matrix_generate(z, M)
        dw = (-1)*np.vstack((np.conj(V_f.T), np.conj(U.T))) @ e
        w_full = np.vstack((w_lut.reshape((nspl,1)), w_fir.reshape((M,1)))).reshape((-1,))
        w_full = w_full - mu*dw
        w_lut, w_fir = w_full[0:nspl], w_full[nspl:nspl + M]
        dataset_part_nmse[k] = sl.nmse(x, d - y)
        print(dataset_part_nmse[k], k)
    print('generated, tx and error energy:', np.var(y), np.var(x), np.var(e))
    generated, z_useless = sl.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated.reshape((-1,)), x, d, dataset_part_nmse

