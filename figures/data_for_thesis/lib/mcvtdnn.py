{}# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:27:53 2021

@author: dWX1065688
"""

import sys
import plot_lib as pl
import scipy.signal as signal
from collections import OrderedDict
import numpy as np
import support_lib as sl
import copy
import torch

class MCVTDNN_FIR:
    '''
        Class MCVTDNN_FIR represents set of adaptation and support functions
        for Multiple Complex-Valued Time-Delayed Neural Network with FIR joint optimization
    '''
    def __init__(self, K: int, Ntaps: int, Ntaps_ds: int, resample_ratio: float, delays_num: int, 
                 nonlin_num: int, internal_range : int, internal_type=int):
        '''
        Parameters
        ----------
        K : int > 0
            Number of neurons in CVTDNN.
        Ntaps : int > 0
            Number of taps in FIR.
        Ntaps_ds : int > 0
            Number of taps in downsampling FIR.
        resample_ratio : float
            Ratio of downsamping after nonlinearity, before filtering
        delays : list
            Includes lists of delays of signal and abs of signal.
            i-th list in list delays corresponds to the delays of i-th CVTDNN in model
            delays[i][0] - signal delay of i-th MCVTDNN
            delays[i][n], n > 0 - abs(signal) delays of i-th CVTDNN
        internal_range : int
            Range of internal linear layer weights: [-internal_range//2; internal_range//2]
        internal_type : type
            Type of internal layer coefficients
            Must be integer of complex
            
        Returns
        -------
        class MCVTDNN_FIR
        
        '''
        # super().__init__()
        assert Ntaps % 2 == 1, "Number of FIR taps must be odd"
        assert Ntaps_ds % 2 == 1, "Number of downsampling FIR taps must be odd"
        assert nonlin_num >= 1, "Number of delay collections must be more or equal 1"
        assert delays_num >= 2, "Number of delays must more or equal 2 (for signal and abs(signal))"
        self.K = K
        self.internal_range = internal_range
        self.Ntaps = Ntaps
        self.Ntaps_ds = Ntaps_ds
        self.nonlin_num = nonlin_num
        self.delays_num = delays_num
        self.resample_ratio = resample_ratio
        if self.resample_ratio < 1:
            self.downsample_ratio = int(1/self.resample_ratio)
        else:
            self.downsample_ratio = int(self.resample_ratio)
        self.internal_type = internal_type
        self.weight = {}
        # Initialize internal Linear Layer
        self.weight.update({'H': np.round(self.internal_range*np.random.rand(self.nonlin_num, self.K, self.delays_num - 1)-\
                                          self.internal_range//2).astype(self.internal_type)})
        self.weight.update({'b_in': np.zeros((self.nonlin_num, self.K), dtype=float).astype(self.internal_type)})
        # Initialize external Linear Layer
        self.weight.update({'a': 2e-2*((np.random.rand(self.nonlin_num, self.K)-0.5) + 1j*((np.random.rand(self.nonlin_num, self.K)-0.5)))})
        self.weight.update({'b_ext': 0+0j})
        # Initialize downsampling FIR taps
        self.weight.update({'fir_ds': np.zeros(self.Ntaps_ds, dtype=complex)})
        self.weight['fir_ds'][self.Ntaps_ds//2] = 1
        # Initialize FIR taps
        self.weight.update({'w': np.zeros(self.Ntaps, dtype=complex)})
        self.weight['w'][self.Ntaps//2] = 1
        return None
    
    def forward(self, x):
        '''
        Parameters
        ----------
        x : numpy.ndarray
            len(delays)*len(delays[0]) x len(x) - dimensional signal, where
            nonlin_num*j-th row includes x with delay delays[j][0]
            nonlin_num*j+i-th row includes abs(x) with delay delays[j][i], where i > 0.

        Returns
        -------
        Model output.

        '''
        z = np.zeros((x.shape[1],), dtype=complex)
        x_init = x.copy()
        for i in range(self.nonlin_num):
            x = x_init[self.delays_num*i, :]
            x_abs = x_init[self.delays_num*i+1:self.delays_num*(i+1), :].real
            g = np.arccos(x_abs)
            b_in = np.expand_dims(self.weight['b_in'][i, :], axis=1)
            phi = self.weight['H'][i, :] @ g + np.kron(b_in, np.ones(x.size))
            psi = np.cos(phi)
            f = self.weight['a'][i, :] @ psi + np.kron(self.weight['b_ext'], np.ones(x.size))
            z += f * x
        z = signal.convolve(z, self.weight['fir_ds'], mode='same')
        z = z[::self.downsample_ratio]
        y = signal.convolve(z, self.weight['w'], mode='same')
        return y
    
    def set_init_weights(self):
        '''
        Function sets initial random weights
        
        Returns
        -------
        None.
        '''
        # Initialize internal Linear Layer
        self.weight['H'] = np.round(self.internal_range*np.random.rand(self.nonlin_num, self.K, self.delays_num - 1)-\
                                    self.internal_range//2).astype(self.internal_type)
        self.weight['b_in'] = np.zeros((self.nonlin_num, self.K), dtype=float).astype(self.internal_type)
        # Initialize external Linear Layer
        self.weight['a'] = 2e-2*((np.random.rand(self.nonlin_num, self.K)-0.5) + 1j*((np.random.rand(self.nonlin_num, self.K)-0.5)))
        self.weight['b_ext'] = 0+0j
        # Downsampling FIR taps don`t change
        # Initialize FIR taps
        self.weight['w'] = np.zeros(self.Ntaps, dtype=complex)
        self.weight['w'][self.Ntaps//2] = 1
        return None
    
    def set_torch_weights(self, weights: dict, set_layers: str):
        '''
        Parameters
        ----------
        weights : dict
            Dictionary of weights from TDNN model.
            Weights are obtained by function
            torch.save(self.state_dict(), path+'weights'),
            Then weights loaded weights preprocessed by function
            utils_nn.weights_prepare(path_to_weights: str)
        set_layers : str
            Flag that shows, for which layer parameters have to be set.
            Flags: 
                'all' - for all layers
                'H' - for internal linear layer weights
                'b_in' - for internal linear layer bias
                'a' - for external linear layer weights
                'b_ext' - for external linear layer bias
                'fir_ds' - for downsampling filter
                'w' - for leakage path filter

        Returns
        -------
        None.
        '''
        if set_layers == 'b_ext':
            self.weight['b_ext'] = weights['nonlin.fc_out.bias']
        if set_layers == 'fir_ds':
            self.weight['fir_ds'] = weights['fir_ds.taps.weight']
        if set_layers == 'w':
            self.weight['w'] = weights['fir.taps.weight']
        return None
      
    def save_weights(self, path: str, weights={}, add_info=''):
        '''
        Function saves weights to path
        
        Parameters
        ----------
        path : str
            Path to save weights.
        weights : dict
            Weights to save. If weights == {}, save weights within the class
        add_info : str
            String to be added to the names of weights. The default is ''.

        Returns
        -------
        None.
        '''
        if weights == {}:
            np.save(path+r'/H'+add_info+'.npy', self.weight['H'])
            np.save(path+r'/b_in'+add_info+'.npy', self.weight['b_in'])
            np.save(path+r'/a'+add_info+'.npy', self.weight['a'])
            np.save(path+r'/b_ext'+add_info+'.npy', self.weight['b_ext'])
            np.save(path+r'/fir_ds'+add_info+'.npy', self.weight['fir_ds'])
            np.save(path+r'/w'+add_info+'.npy', self.weight['w'])
        else:
            np.save(path+r'/H'+add_info+'.npy', weights['H'])
            np.save(path+r'/b_in'+add_info+'.npy', weights['b_in'])
            np.save(path+r'/a'+add_info+'.npy', weights['a'])
            np.save(path+r'/b_ext'+add_info+'.npy', weights['b_ext'])
            np.save(path+r'/fir_ds'+add_info+'.npy', weights['fir_ds'])
            np.save(path+r'/w'+add_info+'.npy', weights['w'])
        return None
  
    def load_weights(self, path: str, add_info=''):
        '''        
        Parameters
        ----------
        path : str
            Path to load weights from.
        add_info : str
            String to be added to the names of weights. The default is ''.

        Returns
        -------
        Loaded weights from the path.
        '''
        weight = {}
        weight.update({'H': np.load(path+r'/H'+add_info+'.npy')})
        weight.update({'b_in': np.load(path+r'/b_in'+add_info+'.npy')})
        weight.update({'a': np.load(path+r'/a'+add_info+'.npy')})
        weight.update({'b_ext': np.load(path+r'/b_ext'+add_info+'.npy')})
        weight.update({'fir_ds': np.load(path+r'/fir_ds'+add_info+'.npy')})
        weight.update({'w': np.load(path+r'/w'+add_info+'.npy')})
        return weight   
    
    def set_weights(self, weight: dict, expand_dim=False):
        '''
        Function updates weights with input dictionary weight
        
        Parameters
        ----------
        weight : dict
            Input weight dictionary to be set in class.
        expand_dim : bool
            Flag, that indicates whether to add dimension to the weights or not.
            expand_dim == True - means weights have shape (dim0, dim1) or (dim0,)
            expand_dim == False - means weights have shape (nonlin_num, dim0, dim1) or (nonlin_num, dim0,)

        Returns
        -------
        None.
        '''
        if expand_dim == True:
            weight['H'] = np.expand_dims(weight['H'], axis=0)
            weight['b_in'] = np.expand_dims(weight['b_in'], axis=0)
            weight['a'] = np.expand_dims(weight['a'], axis=0)
        self.weight['H'] = weight['H']
        self.weight['b_in'] = weight['b_in']
        self.weight['a'] = weight['a']
        self.weight['b_ext'] = weight['b_ext']
        self.weight['fir_ds'] = weight['fir_ds']
        self.weight['w'] = weight['w']
        return None
    
    def set_imp_weights(self, weight):
        '''
        Function sets only important in terms of initial
        conditions optimizations weigths
        
        Parameters
        ----------
        weight : dict
            Full dictionary of weights.

        Returns
        -------
        None.
        '''
        self.weight['H'] = weight['H'].copy()
        return None
    
    def model_dim(self):
        '''
        Returns
        -------
        Number of whole model changeable weights.
        '''
        return self.weight['H'].size + self.weight['b_in'].size + \
               self.weight['a'].size + self.weight['w'].size
    
    def init2vec(self, weight={}):
        '''
        Function provides initial conditions to be
        optimized by init_cond_opt.py means

        Returns
        -------
        Link to the vectorized initial conditions supposed to be optimized.
        '''
        if weight == {}:
            weight = self.weight
            nonlin_num = self.nonlin_num
        else:
            nonlin_num = weight['H'].shape[0]
        block_weight = weight['H'][0, :]
        for i_nonlin in range(1, nonlin_num):
            block_weight = np.hstack((block_weight, weight['H'][i_nonlin, :]))
        return block_weight.flatten('F')
    
    def vec2init(self, weight_vec):
        '''
        Function sets reshapes vectorized initial conditions into the initial
        form and sets this weights to the dictionary of weights

        Returns
        -------
        None.
        '''
        weight_init = np.zeros((self.nonlin_num, self.K, self.delays_num - 1), dtype=complex)
        for i_nonlin in range(self.nonlin_num):
            curr_weight_vec = weight_vec[i_nonlin*self.K*(self.delays_num - 1):(i_nonlin + 1)*self.K*(self.delays_num - 1)]
            weight_init[i_nonlin, :] = curr_weight_vec.reshape(self.delays_num - 1, self.K).T
        self.weight['H'] = weight_init
        return None
    
    def get_init_lim(self):
        '''
        Returns
        -------
        Possible initial points limits.
        '''
        return int(np.round(self.K/2))

class optimizer():
    '''
        Class optimizer represents set of methods for MCVTDNN_FIR adaptation,
        which calculate gradients, hessian etc for Mixed Newton, Newton, Gradient Descent
    '''
    def __init__(self, weight: dict, data: dict, mu=0.5):
        '''
        Parameters
        ----------
        weights : dict
            Link to the dictionary with original weights of MCVTDNN model
        data : dict
            Dictionary with input, target and noise floor signals
        mu : float
            Algorithm step size. The default is 0.5.
            
        Returns
        -------
        Class optimizer
        '''
        # super().__init__()
        # Absorb information from weight and data links
        self.weight = weight
        self.weight_curr = 0
        self.x_full = data['input']
        self.x = 0
        self.d = data['target']
        self.nf = data['noise_floor']
        self.x_sig_len = self.x_full.shape[1]
        self.d_sig_len = self.d.size
        self.K = self.weight['b_in'][0, :].size
        self.tap_num = self.weight['w'].size
        self.nonlin_num = self.weight['H'].shape[0]
        self.delays_num = int(self.x_full.shape[0]/self.nonlin_num)
        # Optimization constants and flags        
        self.mu = mu
        self.init_mu = mu
        self.step_evoluiton = []
        self.loss_evolution = []
        # Model output derivatives
        self.grad_y_w = 0
        self.grad_y_a = 0
        self.grad_y_b = 0
        self.grad_y_h = 0
        self.grad_y_w_full = 0
        self.grad_y_a_full = 0
        self.grad_y_b_full = 0
        self.grad_y_h_full = 0
        self.grad_y = 0
        # MSE derivative and hessian
        self.grad = 0
        self.hessian = 0
        self.inv_hessian = 0
        # Additional useful objects
        self.y = 0
        self.V = 0
        self.x_abs = 0
        self.g = 0
        self.phi = 0
        self.psi = 0
        self.X = 0
        self.F = 0
        self.A = 0
        self.G = 0  
        # Whole weight vector
        self.coeffs = np.array([])         
        return None  

    def load_input_data(self, input_data):
        '''
        Function loads input data to the class and sets
        all corresponding objects and variables

        Parameters
        ----------
        data : dict
            Data, that includes input, target and noise floor signals.

        Returns
        -------
        None.
        '''
        self.x_full = input_data
        self.x_sig_len = self.x_full.shape[1]
        self.delays_num = int(self.x_full.shape[0]/self.nonlin_num)
        return None
        
    def load_cost(self, path):
        '''
        Function loads model output, target and noise floor signals
        and calculates cost funciton

        Parameters
        ----------
        path : str
            Folder path with signals arrays.

        Returns
        -------
        Cost function value.
        '''
        model_out = np.load(path + r'/y.npy')
        target_data = np.load(path + r'/d.npy')
        noise_floor = np.load(path + r'/nf.npy')
        return sl.nmse_nf(target_data, target_data - model_out, noise_floor)

    def calc_obj(self):
        '''
        Function calculate objects, important for
        hessian and gradient calculation

        Returns
        -------
        None.
        '''
        x_init = self.x.copy()
        x = x_init[0, :]
        self.x_abs = x_init[1:, :].real
        self.g = np.arccos(self.x_abs)
        b_in = np.expand_dims(self.weight_curr['b_in'], axis=1)
        self.phi = self.weight_curr['H'] @ self.g + np.kron(b_in, np.ones(self.x_sig_len))
        self.psi = np.cos(self.phi)
        self.V = self.psi.T * (np.kron(np.ones((1, self.K)), np.expand_dims(x, axis=1)))
        self.V = sl.fir_filtering_matrix_conv(self.V, self.weight_curr['fir_ds'])
        self.V = self.V[::2, :]
        self.X_K = np.kron(np.ones((1, self.K)), np.expand_dims(x, axis=1))
        self.X_MK = np.kron(np.ones((1, self.K * (self.delays_num - 1))), np.expand_dims(x, axis=1))
        self.F = np.sin(self.phi.T)
        self.A = np.kron(np.ones((self.x_sig_len, 1)), self.weight_curr['a'])
        self.G = self.g.T
        return None

    def hessian_calc(self):
        '''
        Function calculates MSE Hessian
        with respect to the conjugated full weight vector.
        
        Returns
        -------
        None
        '''
        grad_y_H = np.conj(self.grad_y).T
        self.hessian = grad_y_H @ self.grad_y
        return None

    def gradient_calc(self):
        '''
        Function calculates MSE gradient
        with respect to the conjugated full weight vector

        Returns
        -------
        None
        '''
        self.y = self.grad_y_w_full @ self.weight_curr['w']
        e = self.d - self.y
        self.grad = (-1)*np.conj(self.grad_y).T @ e
        return None
    
    def model_deriv(self):
        '''
        Function calculates model output gradient
        with respect to the whole weight vector
        
        Returns
        -------
        None
        '''
        for i_nonlin in range(self.nonlin_num):
            self.x = self.x_full[self.delays_num*i_nonlin:self.delays_num*(i_nonlin+1), :]
            self.weight_curr = self.extract_curr_weight(indx_nonlin=i_nonlin)
            self.calc_obj()
            self.fir_weight_deriv()
            self.external_weight_deriv()
            self.internal_bias_deriv()
            self.internal_weight_deriv()
            if i_nonlin == 0:
                self.grad_y_w_full = self.grad_y_w.copy()
                self.grad_y_a_full = self.grad_y_a.copy()
                self.grad_y_b_full = self.grad_y_b.copy()
                self.grad_y_h_full = self.grad_y_h.copy()
            else:
                self.grad_y_w_full += self.grad_y_w
                self.grad_y_a_full = np.hstack((self.grad_y_a_full, self.grad_y_a))
                self.grad_y_b_full = np.hstack((self.grad_y_b_full, self.grad_y_b))
                self.grad_y_h_full = np.hstack((self.grad_y_h_full, self.grad_y_h))
        self.grad_y = np.hstack((self.grad_y_w_full, self.grad_y_a_full, self.grad_y_b_full, self.grad_y_h_full))
        return None

    def extract_curr_weight(self, indx_nonlin):
        '''
        Parameters
        ----------
        indx_nonlin : int
            Index of current CVTDNN, which weights are supposed to be extracted.
        
        Returns
        -------
        Extracted weight of current (indx_nonlin) CVTDNN from weight dictionary.
        '''
        weight = {}
        weight.update({'H': self.weight['H'][indx_nonlin, :]})
        weight.update({'b_in': self.weight['b_in'][indx_nonlin, :]})
        weight.update({'a': self.weight['a'][indx_nonlin, :]})
        weight.update({'b_ext': self.weight['b_ext']})
        weight.update({'fir_ds': self.weight['fir_ds']})
        weight.update({'w': self.weight['w']})
        return weight

    def fir_weight_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output
        with respect to the weights of FIR.
        
        Returns
        -------
        None
        '''
        f = self.V @ self.weight_curr['a'] + np.kron(self.weight_curr['b_ext'], np.ones(self.d_sig_len))    
        self.grad_y_w = sl.fir_matrix_generate(f, self.weight_curr['w'].size)
        return None
    
    def external_weight_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output 
        with respect to the weights of external linear layer.
        
        Returns
        -------
        None
        '''
        self.grad_y_a = sl.fir_filtering_matrix_conv(self.V, self.weight_curr['w'])  
        return None
 
    def internal_bias_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output 
        with respect to the bias of internal linear layer.

        Returns
        -------
        None
        '''
        # Derivative of downsample filter input by internal layer weights
        z_deriv = -1*self.X_K * (self.A * self.F)
        # Filter z_deriv by DS-filter, decimate by 2, dilter by FIR weights
        z_deriv_f_ds = sl.fir_filtering_matrix_conv(z_deriv, self.weight_curr['fir_ds'])
        z_deriv_ds = z_deriv_f_ds[::2, :]
        self.grad_y_b = sl.fir_filtering_matrix_conv(z_deriv_ds, self.weight_curr['w'])
        return None   
 
    def internal_weight_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output 
        with respect to the weights of internal linear layer.

        Returns
        -------
        None
        '''
        G_wave = np.kron(self.G, np.ones((1, self.K)))
        F_wave = np.kron(np.ones((1, self.delays_num - 1)), self.A * self.F)
        # Derivative of downsample filter input by internal layer weights
        z_deriv = -1*self.X_MK * (G_wave * F_wave)
        # Filter z_deriv by DS-filter, decimate by 2, dilter by FIR weights
        z_deriv_f_ds = sl.fir_filtering_matrix_conv(z_deriv, self.weight_curr['fir_ds'])
        z_deriv_ds = z_deriv_f_ds[::2, :]
        self.grad_y_h = sl.fir_filtering_matrix_conv(z_deriv_ds, self.weight_curr['w'])
        return None
    
    def parts2whole(self):
        '''        
        Function calculates coefficients stacked into one vector,
        coeffs = (w, a, b_in, vec(H))

        Returns
        -------
        None
        '''
        self.coeffs = self.weight['w'].tolist()
        self.coeffs.extend(self.weight['a'].reshape(self.nonlin_num*self.K,))
        self.coeffs.extend(self.weight['b_in'].reshape(self.nonlin_num*self.K,))
        for i in range(self.nonlin_num):
            self.coeffs.extend(self.matr2vec(self.weight['H'][i, :]))
        self.coeffs = np.array(self.coeffs)
        return None
  
    def whole2parts(self):
        '''
        Function divides stacked coefficients vector:
            coeffs = (w, a, b_in, vec(H))
        into appropriate parts

        Returns
        -------
        None.
        '''
        self.weight['w'] = self.coeffs[:self.tap_num]
        self.weight['a'] = self.coeffs[self.tap_num:self.tap_num+self.K*self.nonlin_num]
        self.weight['a'] = self.weight['a'].reshape(self.nonlin_num, self.K)
        self.weight['b_in'] = self.coeffs[self.tap_num+self.K*self.nonlin_num:self.tap_num+2*self.K*self.nonlin_num]
        self.weight['b_in'] = self.weight['b_in'].reshape(self.nonlin_num, self.K)
        h = self.coeffs[self.tap_num+2*self.K*self.nonlin_num:]
        for i in range(self.nonlin_num):
            vec = h[self.K*(self.delays_num-1)*i:self.K*(self.delays_num-1)*(i+1)]
            self.weight['H'][i, :] = self.vec2matr(vec, self.K, (self.delays_num-1))
        return None
    
    def matr2vec(self, matr):
        '''
        Parameters
        ----------
        matr : numpy.ndarray
            Matrix to be vectorized.

        Returns
        -------
        Vectorized matrix matr
        '''
        return matr.flatten('F')
    
    def vec2matr(self, vec, dim0, dim1):
        '''
        Parameters
        ----------
        vec : numpy.ndarray
            Vector to be reshaped into matrix,
            vector must have size (dim0*dim1,)
        dim0 : int
            First dimension of supposed matrix
        dim1 : int
            Second dimension of supposed matrix

        Returns
        -------
        Matrix with size (dim0, dim1), created from vector vec
        '''
        return vec.reshape(dim1, dim0).T  
    
    def get_step_evol(self):
        '''
        Returns
        -------
        Evolution of step mu in optimizer.
        '''
        return np.array(self.step_evoluiton)
    
    def set_step_evol(self, step):
        '''
        Increases list, that contains step mu evolution

        Returns
        -------
        None.
        '''
        self.step_evoluiton.append(step)
        return None
    
    def get_loss_evol(self):
        '''
        Returns
        -------
        Evolution of loss.
        '''
        return np.array(self.loss_evolution)
    
    def set_loss_evol(self, loss):
        '''
        Increases list, that contains loss evolution

        Returns
        -------
        None.
        '''
        self.loss_evolution.append(loss)
        return None

class Mixed_Newton(optimizer):
    '''
        Class optimizer represents set of methods 
        for training MCVTDNN_FIR with Mixed Newton
    '''
    def __init__(self, weight: dict, data: dict, train_mode='simple', mu=0.5):
        '''
        Constructor
        
        Parameters
        ----------
        train_mode : str
            Flag that shows, which learning strategy is chosen
            train_mode == 'simple' - simple Mixed Newton: mu - fixed
            train_mode == 'damped' - Damped Mixed Newton
            
        Returns
        -------
        Class Mixed_Newton
        '''
        optimizer.__init__(self, weight, data, mu)
        self.train_mode = train_mode
        return None
    
    def step_vec(self):
        '''
        Returns
        -------
        Step vector: inverted loss hessian multiplied by the loss gradient
        '''
        self.parts2whole()
        self.model_deriv()
        self.hessian_calc()
        self.inv_hessian = np.linalg.pinv(self.hessian, rcond=1e-8)
        self.gradient_calc()
        return self.inv_hessian @ self.grad
    
    def step(self):
        '''
        Function represents Mixed Newton step

        Returns
        -------
        None
        '''
        step = self.mu*self.step_vec()
        self.coeffs -= step
        self.whole2parts()
        return None