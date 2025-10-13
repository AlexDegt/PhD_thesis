# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:59:45 2021

@author: dWX1065688
"""

import support_lib as sl
import plot_lib as pl
import scipy.signal as signal
import numpy as np
import alex_degt_support_lib as sup

'''
        Function returns updated coefficients of Stochastic Gradient Descent
    using following optimizers:
      - Momentum
      - Nesterov acceleration (Nesterov)
      - Adadelta
      - RMSprop
      - Adam
      - Nesterov acceleration + Adam (Nadam)
      - AMSGrad
        Parameter 'opt' requires name of optimizer. opt == 'Simple' returns
    updated coefficients without optimizer
        param must contain all objects required by current algorithm. Type of param - tuple
        For example:
    w: current coefficients value
    dw: w[n] = w[n-1] + dw. In simple case dw = mu*grad
    For opt == 'Momentum' param = (momentum, w[n-1]) such that w[n] = momentum*w[n-1] + dw
    norm: coefficient to divied fradient with. For example norm = |x|^2 -> grad = grad/|x|^2
        To make Nesterov acceleration input dw must be: dw[n] = mu*grad[n], where grad[n] calulated at
    w[n - 1] - gamma*dw[n - 1]
        
    
'''
def sgd_opt_update(param, opt = 'Simple', norm = 1):
    '''
        Initialization
    '''
    eps = 1e-8
    if (opt == 'Simple'):
        # Init
        w = param[0]
        grad = param[1]
        mu = param[2]
        # Update
        w = w + (mu*grad/norm)
        return w, (mu/norm)
    elif (opt == 'Momentum'):
        # Init
        w = param[0]
        grad = param[1]
        mu = param[2]
        moment = param[3]
        dw_prev = param[4]
        # Update
        dw = moment*dw_prev - (mu*grad/norm)
        w = w - dw
        return w, dw_prev, (mu/norm)
#    elif (opt == 'Adadelta'):
        # Init
        # Update
    elif (opt == 'RMSprop'):
        # Init
        w = param[0]
        grad = param[1]
        eta = param[2]
        gamma = param[3]
        grad_sqr_aver_prev = param[4]
        # Update
        grad_sqr_aver = gamma*grad_sqr_aver_prev + (1 - gamma)*(np.sum(np.abs(grad)**2))
        mu = (eta/np.sqrt(grad_sqr_aver + eps))
        w = w + mu*grad
        return w, grad_sqr_aver, (mu/norm)
    elif (opt == 'Adam'):
        # Init
        w = param[0]
        grad = param[1]
        eta = param[2]
        beta1 = param[3]
        beta2 = param[4]
        grad_sqr_aver_prev = param[5]
        grad_aver_prev = param[6]
        t = param[7]
        # Update
        grad_aver = beta1*grad_aver_prev + (1 - beta1)*grad
        grad_sqr_aver = beta2*grad_sqr_aver_prev + (1 - beta2)*(np.sum(np.abs(grad)**2))
        grad_aver = grad_aver/(1 - beta1**(t + 1))
        grad_sqr_aver = grad_sqr_aver/(1 - beta2**(t + 1))
        mu = (eta/(np.sqrt(grad_sqr_aver) + eps))
        w = w + mu*grad_aver
        return w, grad_sqr_aver, grad_aver, (mu/norm)
    elif (opt == 'Nadam'):
        # Init
        w = param[0]
        grad = param[1]
        eta = param[2]
        beta1 = param[3]
        beta2 = param[4]
        grad_sqr_aver_prev = param[5]
        grad_aver_prev = param[6]
        t = param[7]
        # Update
        grad_aver = beta1*grad_aver_prev + (1 - beta1)*grad
        grad_sqr_aver = beta2*grad_sqr_aver_prev + (1 - beta2)*(np.sum(np.abs(grad)**2))
        grad_aver = grad_aver/(1 - (beta1**(t + 1)))
        grad_sqr_aver = grad_sqr_aver/(1 - (beta2**(t + 1)))
        mu = (eta/(np.sqrt(grad_sqr_aver) + eps))
        w = w + mu*(beta1*grad_aver + grad*(1 - beta1)/(1 - beta1**(t + 1)))
        return w, grad_sqr_aver, grad_aver, (mu/norm)
    elif (opt == 'AMSGrad'):
        # Init
        w = param[0]
        grad = param[1]
        eta = param[2]
        beta1 = param[3]
        beta2 = param[4]
        grad_sqr_aver_prev = param[5]
        grad_aver_prev = param[6]
        t = param[7]
        # Update
        grad_aver = beta1*grad_aver_prev + (1 - beta1)*grad
        grad_sqr_aver = beta2*grad_sqr_aver_prev + (1 - beta2)*(np.sum(np.abs(grad)**2))
        grad_sqr_aver = np.max([grad_sqr_aver, grad_sqr_aver_prev])
        mu = (eta/(np.sqrt(grad_sqr_aver) + eps))
        w = w + mu*grad_aver
        return w, grad_sqr_aver, grad_aver, (mu/norm)
        
        