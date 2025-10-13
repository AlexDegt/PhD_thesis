# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:36:57 2023

@author: dWX1065688
"""

import sys
import numpy as np
import copy
import support_lib as sl
from time import perf_counter
import utils_nn as utils

def train(net, data, optimizer, track=True, results_path=None):
    '''
    Train model with optimizers written from the scratch

    Parameters
    ----------
    net : optional
        Model class
    data : dict
        Dictionary data must include 3 keys:
        'input', value: numpy.ndarray with input signal
        'target', value: numpy.ndarray with target signal
        'noise_floor', value: numpy.ndarray with noise floor signal
    optimizer : optional
        Optimizer class, with the current model optimization toolkit
    track : bool
        Flag that shows, whether to show intermediate
        performance or not

    Returns
    -------
    None.
    '''
    # Data preparation
    x = data['input']
    d = data['target']
    nf = data['noise_floor']
    optimizer.weight = net.weight
    # Training process
    if optimizer.train_mode == 'simple':
        for epoch in range(optimizer.epoch_num):
            optimizer.step()
            y = net.forward(x)
            NMSE = sl.nmse_nf(d, d - y, nf)
            optimizer.set_step_evol(optimizer.mu)
            optimizer.set_loss_evol(NMSE)
            if track == True:
                print(f'Epoch {epoch}, NMSE = {NMSE} dB')
    if optimizer.train_mode == 'AdaHessian':
        delta_NMSE = 1 # Any number higher, than 0.0001
        epoch = 0
        end_flag = 0
        optimizer.step_evoluiton = []
        optimizer.loss_evolution = []
        beta1 = 0.9
        beta2 = 0.999
        # while delta_NMSE > 0.0001 or end_flag != 3:
        sample_size = 400
        sample_num = d.size - 2*sample_size
        # d = d[:-sample_size]
        # nf = nf[:-sample_size]
        # x = x[:, :-2*sample_size]
        while epoch <= 2000:
        # for epoch in range(optimizer.epoch_num): 
        # for epoch in range(optimizer.epoch_num):
            # Calculate loss before step
            m = np.zeros((net.model_dim(),), dtype=complex)
            v = 0+0j
            y = net.forward(x)
            NMSE_bs = sl.nmse_nf(d, d - y, nf)
            t1 = perf_counter()
            for sample in range(sample_num):
                t11 = perf_counter()
                optimizer.d = d[sample:sample+sample_size]
                optimizer.nf = nf[sample:sample+sample_size]
                optimizer.x_full = x[:, sample:sample+2*sample_size]
                optimizer.x_sig_len = 2*sample_size
                optimizer.d_sig_len = sample_size
                # Calculate Hessian and Gradient for AdaHessian
                optimizer.parts2whole()
                optimizer.model_deriv()
                optimizer.hessian_calc()
                optimizer.gradient_calc()
                m = beta1*m + (1 - beta1)*optimizer.grad
                v = beta2*v + (1 - beta2)*np.sum(np.conj(optimizer.grad).T @ optimizer.grad)
                # v = beta2*v + (1 - beta2)*np.sum(np.conj(optimizer.hessian).T @ optimizer.hessian)
                m /= (1 - beta1**(epoch + 10))
                v /= np.sqrt(1 - beta2**(epoch + 1000))
                # optimizer.inv_hessian = np.sqrt(np.linalg.pinv(h, rcond=1e-8))
                # Optimizer step
                step = (optimizer.mu/v)*m
                optimizer.coeffs -= step
                optimizer.whole2parts()  
                optimizer.set_step_evol(optimizer.mu/v)
                t22 = perf_counter()
                # print(t22 - t11)
            t2 = perf_counter()
            # Calculate loss after step
            y = net.forward(x)
            NMSE_as = sl.nmse_nf(d, d - y, nf)
            delta_NMSE = np.abs(NMSE_bs - NMSE_as)
            NMSE = sl.nmse_nf(d, d - y, nf)
            optimizer.set_loss_evol(NMSE)
            if delta_NMSE <= 0.0001:
                end_flag += 1
            else:
                end_flag = 0
            if track == True:
                print(f'Epoch {epoch}, mu = {optimizer.mu/v}, NMSE = {NMSE} dB, time elapsed: {t2-t1} s')
            epoch += 1
    if optimizer.train_mode == 'damped':
        delta_NMSE = 1 # Any number higher, than 0.0001
        epoch = 0
        converge_flag = 0
        optimizer.step_evoluiton = []
        optimizer.loss_evolution = []
        while (delta_NMSE > 0.001 or optimizer.mu != 1 or converge_flag != 2):
        # while epoch <= 0:
        # for epoch in range(optimizer.epoch_num):
            command_flag = utils.check_command_file(path=results_path)
            if command_flag == utils.SKIP_FLAG:
                break
            # BS - means "before step" 
            y = net.forward(x)
            weight_bs = copy.deepcopy(net.weight)
            NMSE_bs = sl.nmse_nf(d, d - y, nf)
            # Optimizer step (AS - after step)
            step_vec = optimizer.step_vec()
            optimizer.coeffs -= optimizer.mu*step_vec
            # return weight_bs
            optimizer.whole2parts()
            y = net.forward(x)
            NMSE_as = sl.nmse_nf(d, d - y, nf)
            if NMSE_as <= NMSE_bs:
                optimizer.mu *= 2
                if optimizer.mu > 1:
                    optimizer.mu = 1
            else:
                while NMSE_as > NMSE_bs:
                    optimizer.weight = copy.deepcopy(weight_bs)
                    net.weight = optimizer.weight
                    optimizer.parts2whole()
                    optimizer.mu /= 1.5
                    optimizer.coeffs -= optimizer.mu*step_vec
                    optimizer.whole2parts()
                    y = net.forward(x)
                    NMSE_as = sl.nmse_nf(d, d - y, nf)
                    if track == True:
                        print(f'mu = {optimizer.mu}, epoch {epoch}, diverges, NMSE_as = {NMSE_as} dB')
            delta_NMSE = np.abs(NMSE_bs - NMSE_as)
            NMSE = sl.nmse_nf(d, d - y, nf)
            optimizer.set_step_evol(optimizer.mu)
            optimizer.set_loss_evol(NMSE)
            if delta_NMSE <= 0.0001 and optimizer.mu == 1:
                converge_flag += 1
            else:
                converge_flag = 0
            if track == True:
                print(f'Epoch {epoch}, mu = {optimizer.mu}, NMSE = {NMSE} dB')
            epoch += 1
        optimizer.mu = optimizer.init_mu