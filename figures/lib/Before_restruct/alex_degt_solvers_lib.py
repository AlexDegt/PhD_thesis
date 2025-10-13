# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:27:53 2021

@author: dWX1065688
"""

import support_lib as sl
import plot_lib as pl
import scipy.signal as signal
import numpy as np
import alex_degt_support_lib as sup
import alex_degt_optimizers as optim
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

def damped_newton_dynam_solver(x, d, w_lut, w_fir, mu, sym_iter_num, pwr = 2, block_num = 128):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    block_len = np.floor(N/block_num).astype(int)
    e_dyn = np.zeros((block_len*block_num*sym_iter_num,), dtype='complex128')
    dataset_part_nmse = np.zeros(block_num*sym_iter_num)
    alpha = 0.0045
    for i in range(sym_iter_num):     
        for kk in range(0, block_num):
            x_block = x[kk*block_len:(kk+1)*block_len]
            d_block = d[kk*block_len:(kk+1)*block_len]
            V = sup.lut_matrix_generate(x_block, nspl, pwr)
            y_old, z = sup.hammerstein_forward(M, w_lut, w_fir, V)
            e_old = d_block - y_old
            dw_lut, dw_fir = get_delta_newton(w_lut, w_fir, V, z, e_old, M, nspl)
            w_lut_tmp = w_lut + mu*dw_lut
            w_fir_tmp = w_fir + mu*dw_fir
            y_new, z_useless = sup.hammerstein_forward(M, w_lut_tmp, w_fir_tmp, V)
            e_new = d_block - y_new
            nmse_old = sup.nmse_rt(x[24:-24], e_old[24:-24])
    #        nmse_old = sl.nmse(x_block*signal.tukey(np.size(x_block), alpha), e_old*signal.tukey(np.size(e_old), alpha))
            error = nmse_old - sup.nmse_rt(x[24:-24], e_new[24:-24])
    #        error = nmse_old - sl.nmse(x_block*signal.tukey(np.size(x_block), alpha), e_new*signal.tukey(np.size(e_new), alpha))
            if (error >= 0):
                mu *= 2
                if mu > 1:
                    mu = 1
                w_lut, w_fir = w_lut_tmp, w_fir_tmp
                e_dyn[(i*block_num+kk)*block_len:(i*block_num+kk+1)*block_len] = e_new
            else:
                flag = True
                while flag:
                    mu /= 1.5
                    w_lut_tmp_ = w_lut + mu*dw_lut
                    w_fir_tmp_ = w_fir + mu*dw_fir
                    y_diverge, z_useless = sup.hammerstein_forward(M, w_lut_tmp_, w_fir_tmp_, V)
                    e_diverge = d_block - y_diverge
    #                error = nmse_old - sl.nmse(x_block*signal.tukey(np.size(x_block), alpha), e_diverge*signal.tukey(np.size(e_diverge), alpha))
                    error = nmse_old - sup.nmse_rt(x[24:-24], e_diverge[24:-24])
                    if error >= 0:
                        flag = False
                        w_lut, w_fir = w_lut_tmp_, w_fir_tmp_
                        e_dyn[(i*block_num+kk)*block_len:(i*block_num+kk+1)*block_len] = e_diverge
#                    print(mu, i, 'diverges')
            e_dyn_part = e_dyn[(i*block_num+kk)*block_len:(i*block_num+kk+1)*block_len]
            dataset_part_nmse[i*block_num+kk] = sup.nmse_rt(x[24:-24], e_dyn_part[24:-24])
#            dataset_part_nmse[i] = sl.nmse(x_block*signal.tukey(np.size(x_block), alpha), e*signal.tukey(np.size(e), alpha))
        print(mu, dataset_part_nmse[i*(block_num+1)-1], i)
    _, z_useless = sup.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, np.zeros(N), x, d, dataset_part_nmse, e_dyn

def damped_newton_solver(x, d, w_lut, w_fir, mu, sym_iter_num, pwr = 2):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sup.lut_matrix_generate(x, nspl, pwr)
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    dataset_part_mse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old, z = sup.hammerstein_forward(M, w_lut, w_fir, V)
        e_old = d - y_old      
        dw_lut, dw_fir = get_delta_newton(w_lut, w_fir, V, z, e_old, M, nspl)
        w_lut_tmp = w_lut + mu*dw_lut
#        w_fir_tmp = w_fir
        w_fir_tmp = w_fir + mu*dw_fir
        y_new, _ = sup.hammerstein_forward(M, w_lut_tmp, w_fir_tmp, V)
        e_new = d - y_new
        nmse_old = sup.nmse(d[24:-24], e_old[24:-24])
        error = nmse_old - sup.nmse(d[24:-24], e_new[24:-24])
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
                y_diverge, _ = sup.hammerstein_forward(M, w_lut_tmp_, w_fir_tmp_, V)
                e_diverge = d - y_diverge
                error = nmse_old - sup.nmse(d[24:-24], e_diverge[24:-24])
                if error >= 0:
                    flag = False
                    w_lut, w_fir = w_lut_tmp_, w_fir_tmp_
                    e = e_diverge
                print(mu, i, 'diverges')
        dataset_part_nmse[i] = sup.nmse(d[24:-24], e[24:-24])
        dataset_part_mse[i] = sup.mse(d[24:-24] - e[24:-24])
        print(i, mu, 'NMSE:', dataset_part_nmse[i])
    generated, _ = sup.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated, x, d, dataset_part_nmse, dataset_part_mse, e

def damped_newton_solver_gain(x, d, w_lut, w_fir, gain, curr_sig, mu, sym_iter_num, pwr = 2):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    stype_num = np.size(gain)
    V = sup.lut_matrix_generate(x, nspl, pwr)
    Vg = np.zeros((N, stype_num), dtype = 'complex128')
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    for i in range(sym_iter_num):
        y_old, z = sup.hammerstein_forward(M, w_lut, w_fir, V)
        Vg[:, curr_sig] = y_old
        y_old *= gain[curr_sig]
        e_old = d - y_old      
        dw_lut, dw_fir, dgain = get_delta_newton_gain(w_lut, w_fir, V, z, Vg, gain, curr_sig, e_old, M, nspl, stype_num)
        w_lut_tmp = w_lut + mu*dw_lut
        w_fir_tmp = w_fir + mu*dw_fir
        gain_tmp = gain + mu*dgain
        y_new, _ = sup.hammerstein_forward(M, w_lut_tmp, w_fir_tmp, V)
        y_new *= gain_tmp[curr_sig]
        e_new = d - y_new
        nmse_old = sl.nmse(d[24:-24], e_old[24:-24])
        error = nmse_old - sl.nmse(d[24:-24], e_new[24:-24])
        if (error >= 0):
            mu *= 2
            if mu > 1:
                mu = 1
            w_lut, w_fir, gain = w_lut_tmp, w_fir_tmp, gain_tmp
            e = e_new
        else:
            flag = True
            while flag:
                mu /= 1.5
                w_lut_tmp_ = w_lut + mu*dw_lut
                w_fir_tmp_ = w_fir + mu*dw_fir
                gain_tmp_ = gain + mu*dgain
                y_diverge, _ = sup.hammerstein_forward(M, w_lut_tmp_, w_fir_tmp_, V)
                y_diverge *= gain_tmp_[curr_sig]
                e_diverge = d - y_diverge
                error = nmse_old - sl.nmse(d[24:-24], e_diverge[24:-24])
                if error >= 0:
                    flag = False
                    w_lut, w_fir, gain = w_lut_tmp_, w_fir_tmp_, gain_tmp_
                    e = e_diverge
                print(mu, i, 'diverges')
        dataset_part_nmse[i] = sl.nmse(d[24:-24], e[24:-24])
        print(mu, dataset_part_nmse[i], i, gain[curr_sig])
    generated, _ = sup.hammerstein_forward(M, w_lut, w_fir, V)
    generated *= gain[curr_sig]
    return w_lut, w_fir, gain, generated, x, d, dataset_part_nmse, e

def get_delta_newton_gain(w_lut, w_fir, V, z, Vg, gain, curr_sig, e, M, nspl, stype_num):
    V_f = gain[curr_sig]*sup.fir_filtering_matrix_conv(V, w_fir)
    U = gain[curr_sig]*sup.fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U, Vg))
    V_full_H = np.conj(V_full).T
    dw_full = np.linalg.pinv(V_full_H @ V_full, rcond=1e-6) @ V_full_H @ e
    dw_lut, dw_fir, dgain = dw_full[0:nspl], dw_full[nspl:nspl + M], dw_full[nspl + M:nspl + M + stype_num]
    return dw_lut, dw_fir, dgain

def get_delta_newton(w_lut, w_fir, V, z, e, M, nspl):
    V_f = sup.fir_filtering_matrix_conv(V, w_fir)
    U = sup.fir_matrix_generate(z, M)
    V_full = np.hstack((V_f, U))
    V_full_H = np.conj(V_full).T
    dw_full = np.linalg.pinv(V_full_H @ V_full, rcond=1e-6) @ V_full_H @ e
    dw_lut, dw_fir = dw_full[0:nspl], dw_full[nspl:nspl + M]
    return dw_lut, dw_fir

def update_coef_newton(w_lut, w_fir, V, z, e, M, nspl, mu):
    V_f = sup.fir_filtering_matrix_conv(V, w_fir)
    U = sup.fir_matrix_generate(z, M)
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
    U = sup.fir_matrix_generate(x, M)
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
        dataset_part_nmse[i] = sl.nmse(x[24:-24], e[24:-24])
        print(mu, dataset_part_nmse[i], i)
    generated = U @ w
    return w, generated, x, d, dataset_part_nmse, e

def sgd_solver(x, d, w_lut, w_fir, sym_iter_num, opt, param_input, norm):
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    model_centre = np.floor(M/2).astype(int)
    end = N - model_centre - 1
    ind = np.arange(model_centre, end)
    V = sup.lut_matrix_generate(x, nspl, pwr = 2)
    y = np.zeros((N,), dtype='complex128')
    e = np.zeros((N,), dtype='complex128')
    Q = np.zeros((M, nspl), dtype='complex128')
    dataset_part_nmse = np.zeros((sym_iter_num))
    grad_sqr_lut = 0
    grad_sqr_fir = 0
    grad_aver_lut = np.zeros((nspl,), dtype='complex128')
    grad_aver_fir = np.zeros((M,), dtype='complex128')
    mu_mem_lut_part = np.zeros((sym_iter_num))
    mu_mem_fir_part = np.zeros((sym_iter_num))
    dw_lut_prev = np.zeros((nspl,), dtype='complex128')
    dw_fir_prev = np.zeros((M,), dtype='complex128')
    for k in range(sym_iter_num):
        for i in ind:
            Q = V[i - model_centre:i + model_centre + 1, :]
            Q = Q[::-1]
            QTw = (Q.T @ w_fir)
            y[i] = w_lut.T @ QTw
            e[i] = d[i] - y[i]
            if np.abs(e[i]) > 1e+10:
                print('The algorithm diverges, try decreasing the step')
                break
            
            grad_lut = np.conj(QTw) * e[i]
            print(grad_lut.shape)
            grad_fir = np.conj(Q @ w_lut) * e[i]
            if (opt == 'Simple'):
                mu = param_input[0]
                div = param_input[1]
                if (i == 0): mu /= div
                param = (w_lut, grad_lut, mu)
                w_lut, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'Simple', norm = 1)
                param = (w_fir, grad_fir, mu)
                w_fir, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'Simple', norm = 1)
            elif (opt == 'Momentum'):
                mu = param_input[0]
                moment = param_input[1]
                param = (w_lut, grad_lut, mu, moment, dw_lut_prev)
                w_lut, dw_lut_prev, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'Momentum', norm = 1)
                param = (w_fir, grad_fir, mu, moment, dw_fir_prev)
                w_fir, dw_fir_prev, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'Momentum', norm = 1)
            elif (opt == 'RMSprop'):
                eta = param_input[0]
                gamma = param_input[1]
                param = (w_lut, grad_lut, eta, gamma, grad_sqr_lut)
                w_lut, grad_sqr_lut, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'RMSprop', norm = 1)
                param = (w_fir, grad_fir, eta, gamma, grad_sqr_fir)
                w_fir, grad_sqr_fir, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'RMSprop', norm = 1)
            elif (opt == 'Adam'):
                eta = param_input[0]
                beta1 = param_input[1]
                beta2 = param_input[2]
                param = (w_lut, grad_lut, eta, beta1, beta2, grad_sqr_lut, grad_aver_lut, i)
                w_lut, grad_sqr_lut, grad_aver_lut, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'Adam', norm = 1)
                param = (w_fir, grad_fir, eta, beta1, beta2, grad_sqr_fir, grad_aver_fir, i)
                w_fir, grad_sqr_fir, grad_aver_fir, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'Adam', norm = 1)
            elif (opt == 'Nadam'):
                eta = param_input[0]
                beta1 = param_input[1]
                beta2 = param_input[2]
                param = (w_lut, grad_lut, eta, beta1, beta2, grad_sqr_lut, grad_aver_lut, i)
                w_lut, grad_sqr_lut, grad_aver_lut, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'Nadam', norm = 1)
                param = (w_fir, grad_fir, eta, beta1, beta2, grad_sqr_fir, grad_aver_fir, i)
                w_fir, grad_sqr_fir, grad_aver_fir, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'Nadam', norm = 1)
            elif (opt == 'AMSGrad'):
                eta = param_input[0]
                beta1 = param_input[1]
                beta2 = param_input[2]
                param = (w_lut, grad_lut, eta, beta1, beta2, grad_sqr_lut, grad_aver_lut, i)
                w_lut, grad_sqr_lut, grad_aver_lut, mu_mem_lut_part[k] = optim.sgd_opt_update(param, opt = 'AMSGrad', norm = 1)
                param = (w_fir, grad_fir, eta, beta1, beta2, grad_sqr_fir, grad_aver_fir, i)
                w_fir, grad_sqr_fir, grad_aver_fir, mu_mem_fir_part[k] = optim.sgd_opt_update(param, opt = 'AMSGrad', norm = 1)
         
        y, _ = sup.hammerstein_forward(M, w_lut, w_fir, V)
        dataset_part_nmse[k] = sl.nmse(x, d - y.reshape((-1,)))
        print(dataset_part_nmse[k], k)
    print('generated, tx and error energy:', np.var(y), np.var(x), np.var(e))
    generated = np.zeros(N - M, 'complex128')
    for j in range(N - M):
        Q = V[j:j + M, :]
        Q = Q[::-1]
        generated[j] = np.transpose(w_fir) @ (Q @ w_lut)
    x = x[model_centre:N - model_centre - 1]
    d = d[model_centre:N - model_centre - 1]
    return w_lut, w_fir, generated, x, d, dataset_part_nmse, e, mu_mem_lut_part, mu_mem_fir_part

def lms_solver(x, d, w_lut, w_fir, mu_start, sym_iter_num):
    mu = mu_start
    N = np.size(x)
    nspl = np.size(w_lut)
    M = np.size(w_fir)
    V = sup.lut_matrix_generate(x, nspl, pwr = 2)
    y = np.zeros((N,), dtype='complex128')
    e = np.zeros((N,), dtype='complex128')
    dataset_part_nmse = np.zeros(sym_iter_num)
    for k in range(sym_iter_num):
#        if (k > int(sym_iter_num/2) - 1):
#            mu /= 10
        y, z = sup.hammerstein_forward(M, w_lut, w_fir, V)
        y = y.reshape((-1,))
        e = d - y
        if np.var(e) > 1e+10:
            print('The algorithm diverges, try decreasing the step')
            break
        V_f = sup.fir_filtering_matrix_conv(V, w_fir)
        U = sup.fir_matrix_generate(z, M)
        dw = (-1)*np.vstack((np.conj(V_f.T), np.conj(U.T))) @ e
        w_full = np.vstack((w_lut.reshape((nspl,1)), w_fir.reshape((M,1)))).reshape((-1,))
        w_full = w_full - mu*dw
        w_lut, w_fir = w_full[0:nspl], w_full[nspl:nspl + M]
        dataset_part_nmse[k] = sl.nmse(x, d - y)
        print(dataset_part_nmse[k], k)
    print('generated, tx and error energy:', np.var(y), np.var(x), np.var(e))
    generated, z_useless = sup.hammerstein_forward(M, w_lut, w_fir, V)
    return w_lut, w_fir, generated.reshape((-1,)), x, d, dataset_part_nmse

