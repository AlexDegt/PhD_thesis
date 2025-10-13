# -*- coding: utf-8 -*-
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
import torch

class CVTDNN_FIR:
    '''
        Class CVTDNN_FIR represents set of adaptation and support functions
        for Complex-Valued Time-Delayed Neural Network with FIR joint optimization
    '''
    def __init__(self, K: int, Ntaps: int, Ntaps_ds: int, resample_ratio: float, delays=[0, 0], internal_type=int):
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
            Includes delays of signal and abs of signal
            delays[0] - signal delay
            delays[i], i > 0 - abs(signal) delays
        internal_type : type
            Type of internal layer coefficients
            Must be integer of complex
            
        Returns
        -------
        class CVTDNN_FIR
        
        '''
        # super().__init__()
        assert Ntaps % 2 == 1, "Number of FIR taps must be odd"
        assert Ntaps_ds % 2 == 1, "Number of downsampling FIR taps must be odd"
        assert len(delays) >= 2, "Number of delays must more of equal 2 (for signal and abs(signal))"
        self.K = K
        self.Ntaps = Ntaps
        self.Ntaps_ds = Ntaps_ds
        self.delays = delays
        self.delays_num = len(delays)
        self.resample_ratio = resample_ratio
        if self.resample_ratio < 1:
            self.downsample_ratio = int(1/self.resample_ratio)
        else:
            self.downsample_ratio = int(self.resample_ratio)
        self.internal_type = internal_type
        self.weight = {}
        # Initialize internal Linear Layer
        self.weight.update({'H': np.round(1*self.K*np.random.rand(self.K, self.delays_num - 1)-self.K//2).astype(self.internal_type)})
        self.weight.update({'b_in': np.zeros((self.K,), dtype=float)})
        # Initialize external Linear Layer
        self.weight.update({'a': 2e-2*((np.random.rand(self.K,)-0.5) + 1j*((np.random.rand(self.K,)-0.5)))})
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
            len(delays) x len(x) - dimensional signal, where
            0-th row includes x with delay delays[0]
            i-th row includes abs(x) with delay delays[i], where i > 0.

        Returns
        -------
        Model output.

        '''
        x_init = x.copy()
        x = x_init[0, :]
        x_abs = x_init[1:, :].real
        g = np.arccos(x_abs)
        b_in = np.expand_dims(self.weight['b_in'], axis=1)
        phi = self.weight['H'] @ g + np.kron(b_in, np.ones(x.size))
        psi = np.cos(phi)
        f = self.weight['a'] @ psi + np.kron(self.weight['b_ext'], np.ones(x.size))     
        z = f * x
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
        self.weight['H'] = np.round(1*self.K*np.random.rand(self.K, self.delays_num - 1)-self.K//2).astype(self.internal_type)
        self.weight['b_in'] = np.zeros((self.K,), dtype=float)
        # Initialize external Linear Layer
        self.weight['a'] = 2e-2*((np.random.rand(self.K,)-0.5) + 1j*((np.random.rand(self.K,)-0.5)))
        self.weight['b_ext'] = 0+0j
        # Initialize downsampling FIR taps
        self.weight['fir_ds'] = np.zeros(self.Ntaps_ds, dtype=complex)
        self.weight['fir_ds'][self.Ntaps_ds//2] = 1
        # Initialize FIR taps
        self.weight['w'] = np.zeros(self.Ntaps, dtype=complex)
        self.weight['w'][self.Ntaps//2] = 1
        return None
    
    def set_torch_weights(self, weights: dict, set_layers='all'):
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
        if set_layers == 'all':
            self.weight['H'] = weights['nonlin.fc_layers.0.weight']
            self.weight['b_in'] = weights['nonlin.fc_layers.0.bias']
            self.weight['a'] = weights['nonlin.fc_out.weight']
            self.weight['b_ext'] = weights['nonlin.fc_out.bias']
            self.weight['fir_ds'] = weights['fir_ds.taps.weight']
            self.weight['w'] = weights['fir.taps.weight']
        if set_layers == 'H':
            self.weight['H'] = weights['nonlin.fc_layers.0.weight']
        if set_layers == 'b_in':
            self.weight['b_in'] = weights['nonlin.fc_layers.0.bias']
        if set_layers == 'a':
            self.weight['a'] = weights['nonlin.fc_out.weight']
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
    
    def set_weights(self, weight: dict):
        '''
        Function updates weights with input dictionary weight
        
        Parameters
        ----------
        weight : dict
            Input weight dictionary to be set in class.

        Returns
        -------
        None.
        '''
        self.weight['H'] = weight['H']
        self.weight['b_in'] = weight['b_in']
        self.weight['a'] = weight['a']
        self.weight['b_ext'] = weight['b_ext']
        self.weight['fir_ds'] = weight['fir_ds']
        self.weight['w'] = weight['w']
        return None
  
    def weights_torch_prepare(self, out_bias=False):
        '''
        Function creates OrderedDict from model weights.
        Then function reshapes tensors in dictionary
        into the suitable for pytorch scripts shapes
        
        Parameters
        ----------
        out_bias : bool
            Flag that shows, whether to include bias of
            the external layer to the weights dictionary or not
        
        Returns
        -------
        OrderedDict from the TDNN+FIR model reshaped tensor weights.
        '''
        # Create and fill orderes dictionary with tensor weights
        weights = OrderedDict()
        weights.update({'nonlin.fc_layers.0.weight': torch.tensor(self.weight['H'])})
        weights.update({'nonlin.fc_layers.0.bias': torch.tensor(self.weight['b_in'])})
        weights.update({'nonlin.fc_out.weight': torch.tensor(self.weight['a'])})
        if out_bias == True:
            weights.update({'nonlin.fc_out.bias': torch.tensor(self.weight['b_ext'])})
        weights.update({'fir_ds.taps.weight': torch.tensor(self.weight['fir_ds'])})
        weights.update({'fir.taps.weight': torch.tensor(self.weight['w'])})  
        # Reshape tensors
        weights['nonlin.fc_out.weight'] = torch.unsqueeze(weights['nonlin.fc_out.weight'], dim=0)
        if out_bias == True:
            weights['nonlin.fc_out.bias'] = torch.unsqueeze(weights['nonlin.fc_out.bias'], dim=0)
        weights['fir_ds.taps.weight'] = torch.unsqueeze(weights['fir_ds.taps.weight'], dim=0)
        weights['fir_ds.taps.weight'] = torch.unsqueeze(weights['fir_ds.taps.weight'], dim=0)
        weights['fir.taps.weight'] = torch.unsqueeze(weights['fir.taps.weight'], dim=0)
        weights['fir.taps.weight'] = torch.unsqueeze(weights['fir.taps.weight'], dim=0)
        return weights
    
    def save_torch_weights(self, weights: OrderedDict, path='', add_info=''):
        '''
        Function saves OrderedDict weights to the path

        Parameters
        ----------
        weights : OrderedDict
            Model weights preprocessed with function weights_torch_prepare.
        path : str
            Path to save OrderedDict model weights. The default is ''.
        add_info : str
            String to be added to the names of weights. The default is ''.

        Returns
        -------
        None.
        '''
        torch.save(weights, path+'/weights'+add_info)
        return None  
    
    def weight_stack(self, weight={}, Ncopies=1, path='', add_info=''):
        '''        
        Parameters
        ----------
        weight : dict
            Weights to stack. The default is {}.
        Ncopies : int
            Number of times to stack. The default is 1.
        path : str
            Path to load weights to stack. The default is ''.
        add_info : str
            String to be added to the names of weights. The default is ''.

        Returns
        -------
        Function loads weights dictionary with stacked internal linear
        layer weights, biases and stacked external linear layer weights.
        Weights stacked Ncopies times.
        '''
        if weight == {}:
            weight = self.load_weights(path, add_info)
        weight['b_in'] = weight['b_in'].reshape((weight['b_in'].size, 1))
        weight['a'] = weight['a'].reshape((weight['a'].size, 1))
        init_weight_H = weight['H']
        init_weight_b_in = weight['b_in']
        init_weight_a = weight['a']
        for i_copy in range(Ncopies):
            weight['H'] = np.vstack((weight['H'], init_weight_H))
            weight['b_in'] = np.vstack((weight['b_in'], init_weight_b_in))
            weight['a'] = np.vstack((weight['a'], init_weight_a))
        weight['b_in'] = weight['b_in'].reshape((weight['b_in'].size,))
        weight['a'] = weight['a'].reshape((weight['a'].size,))
        return weight

class optimizer():
    '''
        Class optimizer represents set of methods for CVTDNN_FIR adaptation,
        which calculate gradients, hessian etc for Mixed Newton, Newton, Gradient Descent
    '''
    def __init__(self, weight: dict, data: dict, layer_mode='full', mu=0.5, epoch_num=15):
        '''
        Parameters
        ----------
        weights : dict
            Link to the dictionary with original weights of CVTDNN model
        data : dict
            Dictionary with input, target and noise floor signals
        layer_mode : str
            Flag, that shows, which adaptation method is used
            layer_mode == 'ext' - adapt external linear layer and FIR with Mixed Newton
            layer_mode == 'full' - adapt whole model with Mixed Newton (except external bias b_ext)
        mu : float
            Algorithm step size. The default is 0.5.
        epoch_num : int
            Number of algorithm iterations. The default is 15.
            
        Returns
        -------
        Class optimizer
        '''
        # super().__init__()
        # Absorb information from weight and data links
        self.weight = weight
        self.x = data['input']
        self.d = data['target']
        self.nf = data['noise_floor']
        self.x_sig_len = self.x.shape[1]
        self.d_sig_len = self.d.size
        self.K = self.weight['b_in'].size
        self.delays_num = self.x.shape[0]
        # Optimization constants and flags        
        self.layer_mode = layer_mode
        self.mu = mu
        self.epoch_num = epoch_num
        self.step_evoluiton = []
        self.loss_evolution = []
        # Model output derivatives
        self.grad_y_w = 0
        self.grad_y_a = 0
        self.grad_y_b = 0
        self.grad_y_h = 0
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
        b_in = np.expand_dims(self.weight['b_in'], axis=1)
        self.phi = self.weight['H'] @ self.g + np.kron(b_in, np.ones(self.x_sig_len))
        self.psi = np.cos(self.phi)
        self.V = self.psi.T * (np.kron(np.ones((1, self.K)), np.expand_dims(x, axis=1)))
        self.V = sl.fir_filtering_matrix_conv(self.V, self.weight['fir_ds'])
        self.V = self.V[::2, :]
        self.X_K = np.kron(np.ones((1, self.K)), np.expand_dims(x, axis=1))
        self.X_MK = np.kron(np.ones((1, self.K * (self.delays_num - 1))), np.expand_dims(x, axis=1))
        self.F = np.sin(self.phi.T)
        self.A = np.kron(np.ones((self.x_sig_len, 1)), self.weight['a'])
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
        self.model_deriv()
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
        self.model_deriv()
        self.y = self.grad_y_w @ self.weight['w']
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
        if self.layer_mode == 'ext':
            self.fir_weight_deriv()
            self.external_weight_deriv()
            self.grad_y = np.hstack((self.grad_y_w, self.grad_y_a))
        if self.layer_mode == 'full':
            self.fir_weight_deriv()
            self.external_weight_deriv()
            self.internal_bias_deriv()
            self.internal_weight_deriv()
            self.grad_y = np.hstack((self.grad_y_w, self.grad_y_a, self.grad_y_b, self.grad_y_h))
        return None

    def fir_weight_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output
        with respect to the weights of FIR.
        
        Returns
        -------
        None
        '''
        f = self.V @ self.weight['a'] + np.kron(self.weight['b_ext'], np.ones(self.d_sig_len))    
        self.grad_y_w = sl.fir_matrix_generate(f, self.weight['w'].size)
        return None
    
    def external_weight_deriv(self):
        '''
        Function calculates matrix derivative of TDNN output 
        with respect to the weights of external linear layer.
        
        Returns
        -------
        None
        '''
        self.grad_y_a = sl.fir_filtering_matrix_conv(self.V, self.weight['w'])  
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
        z_deriv_f_ds = sl.fir_filtering_matrix_conv(z_deriv, self.weight['fir_ds'])
        z_deriv_ds = z_deriv_f_ds[::2, :]
        self.grad_y_b = sl.fir_filtering_matrix_conv(z_deriv_ds, self.weight['w'])
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
        z_deriv_f_ds = sl.fir_filtering_matrix_conv(z_deriv, self.weight['fir_ds'])
        z_deriv_ds = z_deriv_f_ds[::2, :]
        self.grad_y_h = sl.fir_filtering_matrix_conv(z_deriv_ds, self.weight['w'])
        return None
    
    def parts2whole(self):
        '''        
        Function calculates coefficients stacked into one vector,
        coeffs = (w, a) - for layer_mode == 'ext'
        coeffs = (w, a, b_in, vec(H)) - for layer_mode == 'full'

        Returns
        -------
        None
        '''
        if self.layer_mode == 'ext':
            self.coeffs = self.weight['w'].tolist()
            self.coeffs.extend(self.weight['a'])
            self.coeffs = np.array(self.coeffs)
        if self.layer_mode == 'full':
            self.coeffs = self.weight['w'].tolist()
            self.coeffs.extend(self.weight['a'])
            self.coeffs.extend(self.weight['b_in'])
            self.coeffs.extend(self.matr2vec(self.weight['H']))
            self.coeffs = np.array(self.coeffs)
        return None
  
    def whole2parts(self):
        '''
        Function divides stacked coefficients vector:
            coeffs = (w, a) - for layer_mode == 'ext'
            coeffs = (w, a, b_in, vec(H)) - for layer_mode == 'full'
        into appropriate parts

        Returns
        -------
        None.
        '''
        if self.layer_mode == 'ext':
            self.weight['w'] = self.coeffs[:self.weight['w'].size]
            self.weight['a'] = self.coeffs[self.weight['w'].size:]
        if self.layer_mode == 'full':
            self.weight['w'] = self.coeffs[:self.weight['w'].size]
            self.weight['a'] = self.coeffs[self.weight['w'].size:self.weight['w'].size+self.weight['a'].size]
            self.weight['b_in'] = self.coeffs[self.weight['w'].size+self.weight['a'].size:self.weight['w'].size+self.weight['a'].size+self.weight['b_in'].size]
            h = self.coeffs[self.weight['w'].size+self.weight['a'].size+self.weight['b_in'].size:]
            self.weight['H'] = self.vec2matr(vec=h, dim0=self.K, dim1=self.delays_num - 1)
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
        for training CVTDNN_FIR with Mixed Newton
    '''
    def __init__(self, weight: dict, data: dict, train_mode='simple', layer_mode='full', mu=0.5, epoch_num=15):
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
        optimizer.__init__(self, weight, data, layer_mode, mu, epoch_num)
        self.train_mode = train_mode
        return None
    
    def step_vec(self):
        '''
        Returns
        -------
        Step vector: inverted loss hessian multiplied by the loss gradient
        '''
        self.parts2whole()
        self.calc_obj()
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
            
def train(net, data, optimizer, track=True):
    '''
    Train CVTDNN model

    Parameters
    ----------
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
    # Training process
    if optimizer.train_mode == 'simple':
        for epoch in range(optimizer.epoch_num):
            optimizer.step()
            y = net.forward(x)
            NMSE = sl.nmse_nf(d, d - y, nf)
            if track == True:
                print(f'Epoch {epoch}, NMSE = {NMSE} dB')
    if optimizer.train_mode == 'damped':
        delta_NMSE = 1 # Any number higher, than 0.0001
        epoch = 0
        while delta_NMSE > 0.0001 or optimizer.mu != 1:
        # for epoch in range(optimizer.epoch_num):
            # BS - means "before step" 
            y = net.forward(x)
            weight_bs = net.weight.copy()
            NMSE_bs = sl.nmse_nf(d, d - y, nf)
            # Optimizer step (AS - after step)
            hessian_grad = optimizer.step_vec()
            optimizer.coeffs -= optimizer.mu*hessian_grad
            optimizer.whole2parts()
            y = net.forward(x)
            NMSE_as = sl.nmse_nf(d, d - y, nf)
            if NMSE_as <= NMSE_bs:
                optimizer.mu *= 2
                if optimizer.mu > 1:
                    optimizer.mu = 1
            else:
                while NMSE_as > NMSE_bs:
                    optimizer.weight = weight_bs.copy()
                    net.weight = optimizer.weight
                    optimizer.parts2whole()
                    optimizer.mu /= 1.5
                    optimizer.coeffs -= optimizer.mu*hessian_grad
                    optimizer.whole2parts()
                    y = net.forward(x)
                    NMSE_as = sl.nmse_nf(d, d - y, nf)
                    print(f'mu = {optimizer.mu}, epoch {epoch}, diverges, NMSE_as = {NMSE_as} dB')
            delta_NMSE = np.abs(NMSE_bs - NMSE_as)
            NMSE = sl.nmse_nf(d, d - y, nf)
            optimizer.set_step_evol(optimizer.mu)
            optimizer.set_loss_evol(NMSE)
            print(f'Epoch {epoch}, mu = {optimizer.mu}, NMSE = {NMSE} dB')
            epoch += 1
            
    
            
            