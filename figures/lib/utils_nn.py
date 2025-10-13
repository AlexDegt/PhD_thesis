# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:57:16 2022

@author: dWX1065688
"""

import sys
import os
import numpy as np
import re
import typing as tp
import torch
import torch.nn as nn
import seaborn as sns
from pathlib import Path
from shutil import rmtree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from pathlib import Path
import support_lib as sl
import itertools
from time import perf_counter
from torch.utils.data import Dataset
import torch.nn.utils.prune as prune
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv2d, ComplexLinear

import future, sys, os, datetime, argparse
from typing import List, Tuple
from torch.nn import Sequential, Module, Parameter
import copy

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SKIP_FLAG = 'skip'

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),\
                plt.Line2D([0], [0], color="b", lw=4),\
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def same_value(tensor):
    ''' Function that returns the same tensor '''
    return tensor

def data_prepare(x, y, sig_len, nf=[], shift=0, norm=True):
    '''
        Time shift, normalize and cut data before dataset creation
    '''
    # Time shift
    y = np.roll(y, shift)
    # Normalization
    if norm == True:
        x /= np.max(np.abs(x))
        if nf != []:
            nf /= np.max(np.abs(y))
        y /= np.max(np.abs(y))
    # Cut data samples
    x = x[:sig_len]
    y = y[:sig_len]
    if nf != []:
        nf = nf[:sig_len]
    return x, y, nf

def trial_name_optuna(trial, features, delays, comment):
    '''
        Create simulaition name from the model/algorithm parameters
    '''
    trial_name = ""
    for key, value in trial.params.items():
        trial_name += "{}: {}, ".format(key, value)
    for feature in features:
        trial_name += feature.__name__+", "
    trial_name += 'delay: '
    for delay in delays:
        trial_name += f'{delay}, '
    trial_name += comment
    return trial_name

def dir_save_optuna(trial, features, delays, comment):
    '''
        Create directory to save results
        Name of directory is based on the simulation name

        dir_save - directory to save 
    '''
    trial_name = trial_name_optuna(trial, features, delays, comment)
    dir_save = os.getcwd()+r'/results/'+trial_name
    try:
        os.mkdir(dir_save)
    except:
        print(f'Simulation {trial_name} already exists!')
    return dir_save

def sim_name(parameters):
    sim_name = ''
    parameters_keys = list(parameters.keys())
    parameters_len = len(parameters_keys)
    for i_key, key in enumerate(parameters_keys):
        sim_name += key
        sim_name += ': '
        sim_name += str(parameters[key])
        if i_key != parameters_len - 1:
            sim_name += ', '
    return sim_name

def dir_save(parameters, add_name=r'/', create_dir=True):
    simul_name = sim_name(parameters)
    if add_name != r'/':
        add_name = r'/'+add_name+r'/'
    dir_to_save = os.getcwd()+r'/results'+add_name+simul_name
    try:
        if create_dir == True:
            os.mkdir(dir_to_save)
    except FileNotFoundError:
        print(f'{dir_to_save}: there is now such directory!')
        print('Program is stopped')
        sys.exit()
    except FileExistsError:
        print(f'Simulation {simul_name} already exists!')
    return dir_to_save

def len_nested_list(in_list):
    nested_list_len = 0
    for sublist in in_list:
        assert type(sublist) == list, f'List {in_list} must contain only lists (sublists)'
        sublist_len = len(sublist)
        nested_list_len += sublist_len
    return nested_list_len

def feature2ind(feature, features, delays):
    joint_dict = {}
    for i in range(len(features)):
        joint_dict.update({features[i]: delays[i]})
    indices_len = len(joint_dict[feature])
    joint_dict_keys = list(joint_dict.keys())
    first_ind = 0
    for key in joint_dict_keys:
        if key != feature:
            first_ind += len(joint_dict[key])
        else: 
            break
    indices = list(np.arange(first_ind, first_ind + indices_len))
    return indices

def gain_LS_search(d, x):
    '''
        Function searches complex variable alpha,
        such that: ||d - alpha*x||_2 -> min,
        where x, y - complex valued vectors
    '''
    alpha = (np.conj(x) @ d)/(np.conj(x) @ x)
    return alpha

def norm2(x):
    '''
        Function returns second norm of the tensor x
    '''
    return torch.sum(torch.abs(x)**2)

def norm2_mean(x):
    '''
        Function returns second norm of the tensor x
    '''
    x = torch.mean(torch.abs(x)**2, dim=1)
    x = torch.mean(x)
    return x

def init_matrix(dims):
    '''
        T_{n_0}*T_{n_1}*...*T_{n_{L-1}} = 
        = T_{n_0, n_1,..., n_{L-1}} - Basis functions of 
        (L-1)-dimensional nonlinearity, based on Chebyshev polynomials
        Function returns matrix M that contains all combinations
        of (n_0, n_1, ..., n_{L-1}) indices, where
        0 <= n_0 <= N_0-1
        0 <= n_1 <= N_1-1
        ...
        0 <= n_{L-1} <= N_{L-1}-1
        Function takes N_0, N_1,..., N_{L-1} as an input
    '''
    coeffs = []
    combin_num = 1
    dims_num = len(dims)
    for dim in dims:
        coeffs.append(np.arange(dim).tolist())
        combin_num *= dim
    M = torch.zeros(combin_num, dims_num, device=device, dtype=torch.float)
    i = 0
    for element in itertools.product(*coeffs):
        M[i, :] = 2*torch.tensor(element)
        i += 1
    return M

def dataset_prepare(x, delays):
    '''
    Parameters
    ----------
    x : numpy.ndarray
        N-dimensional model (or NN) input complex signal.
    delays : list
        Includes lists of delays of signal and abs of signal.
        i-th list in list delays corresponds to the delays of i-th CVTDNN in model
        delays[i][0] - signal delay of i-th CVTDNN
        delays[i][n], n > 0 - abs(signal) delays of i-th CVTDNN

    Returns
    -------
    len(delays[0])*len(delays) x len(x) - dimensional signal
    '''
    x_size = len(x)
    nonlin_num = len(delays)
    delays_size = len(delays[0])
    x_prepared = np.zeros((delays_size*nonlin_num, x_size), dtype=complex)
    for i_nonlin in range(nonlin_num):
        x_prepared[delays_size*i_nonlin, :] = np.roll(x, -1*delays[i_nonlin][0])
        for i_delay in range(delays_size):
            if i_delay > 0:
                x_prepared[delays_size*i_nonlin+i_delay, :] = np.abs(np.roll(x, -1*delays[i_nonlin][i_delay]))
    return x_prepared

def weights_unpack(path: str):
    '''
    Parameters
    ----------
    path : str
        Path to the torch weights.

    Returns
    -------
    Tuple with arrays of weights,
    unpacked from pytorch file.
    '''
    weights = torch.load(path, map_location=torch.device(device))
    dict_values = list(weights.values())
    dict_keys = list(weights.keys())
    weights = dict(weights)
    for i_weight, weight in enumerate(dict_values):
        weights[dict_keys[i_weight]] = np.array(weight.tolist())
    return weights

def weights_reshape(weights: list):
    '''
    Parameters
    ----------
    weights : list
        The output of weights_unpack function.
        Weights in weights have sizes from pytorch scripts
        and not suitable for cvtdnn class in cvtdnn.py usage

    Returns
    -------
    Weights which are suitable for cvtdnn class usage.
    '''
    # odict_keys(['nonlin.fc_layers.0.weight', 'nonlin.fc_layers.0.bias', 
    # 'nonlin.fc_out.weight', 'nonlin.fc_out.bias', 'fir_ds.taps.weight', 'fir.taps.weight'])
    # weights_dict['nonlin.fc_layers.0.weight'].shape = (K, len(delays)) - Shape suits
    # weights_dict['nonlin.fc_layers.0.bias'].shape   = (K,) - Shape suits
    # weights_dict['nonlin.fc_out.weight'].shape      = (1, K) - Shape doesn`t suit, should be (K,)
    # weights_dict['nonlin.fc_out.bias'].shape        = (1,) - Shape doesn`t suit, should be scalar
    # weights_dict['fir_ds.taps.weight'].shape        = (1, 1, len(fir_ds)) - Shape doesn`t suit, should be (len(fir_ds),)
    # weights_dict['fir.taps.weight'].shape           = (1, 1, len(w)) - Shape doesn`t suit, should be (len(w),)
    weights['nonlin.fc_out.weight'] = weights['nonlin.fc_out.weight'].reshape(weights['nonlin.fc_out.weight'].size,)
    weights['nonlin.fc_out.bias'] = weights['nonlin.fc_out.bias'][0]
    weights['fir_ds.taps.weight'] = weights['fir_ds.taps.weight'][0, 0, :]
    weights['fir.taps.weight'] = weights['fir.taps.weight'][0, 0, :]
    return weights

def weights_prepare(path: str):
    '''
    Parameters
    ----------
    path : str
        Path to the torch weights.

    Returns
    -------
    Weights which are suitable for cvtdnn class usage.
    '''
    weights = weights_unpack(path)
    weights = weights_reshape(weights)
    return weights

def search_individs(low, high, path):
    '''
    Function searches for the individuals with the performance in
    range [low, high]. Simulation folder with populations is located at the path 'path'.

    Parameters
    ----------
    low : float
        Lower bound of the performance to choose individuals from.
    high : float
        Higher bound of the performance to choose individuals from.
    path : str
        Simulation folder path.
        
    Example:
    -------
    path = r'D:\FEIC\Projects\feic\CVTDNN\results\Genetic\8_neurons_3_TDNN_3_inputs_3_breed_num_test'
    info = utils.search_individs(low=-12.5, high=-11.95, path=path)
    
    Returns
    -------
    Dictionary of information about acceptable individuals.
    Key 'loss', value: list of losses
    Key 'popul', value: list of corresponding population indices
    Key 'individ', value: list of correspond ingindividual indices
    '''
    individ_info = {'loss': [], 'popul': [], 'individ': []}
    p = Path(path)
    dirs_popul = [x for x in p.iterdir() if x.is_dir()]
    dir_popul_num = len(dirs_popul) - 1 # best_individual folder mustn`t be taken into account
    for i_dir_popul in range(dir_popul_num):
        popul_path = path + f'/population_{i_dir_popul}'
        p_popul = Path(popul_path)
        dirs_individ = [x for x in p_popul.iterdir() if x.is_dir()]
        dir_individ_num = len(dirs_individ)
        for i_dir_individ in range(dir_individ_num):
            individ_path = popul_path + f'/individual_{i_dir_individ}'
            model_out_data = np.load(individ_path+r'/y.npy')
            target_data = np.load(individ_path+r'/d.npy')
            noise_floor_data = np.load(individ_path+r'/nf.npy')
            curr_loss = sl.nmse_nf(target_data, target_data - model_out_data, noise_floor_data)
            if curr_loss >= low and curr_loss <= high:
                individ_info['loss'].append(curr_loss)
                individ_info['popul'].append(i_dir_popul)
                individ_info['individ'].append(i_dir_individ)
    return individ_info

def draw_population(path, populs: list, mode='init', nfig=None, ax=None, clf=True):
    '''
    Function draws populations fitnesses from the simulation
    located at the path 'path'.

    Parameters
    ----------.
    path : str
        Simulation folder path.
    populs : list
        List of population indices. Default is None.
    mode : str
        mode == 'init' - draws populations for initial weights search.
        mode == 'delay' - draws populations for delays search.
        Default is 'init'.
    nfig : int
        Figure number. The default is None
    ax : figure
        Figure class. The default is None
    clf : bool
        Flag that shows whether to clear figure or not. The default is True

    Returns
    -------
    None.
    '''
    assert type(populs) is list or type(populs) is np.ndarray, \
        f'Type of the variable \'populs\' is {type(populs)}, but must be list of integers'
    figsize_x = 7
    figsize_y = 5
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)  
    if clf:
        ax.cla()
    ax.set_xlabel('Individual index')
    ax.set_ylabel('NMSE [dB]')
    ax.set_title(f'Populations {populs[0]}-{populs[-1]}', fontsize=20)
    ax.grid(True)
    if mode == 'init':
        for i, indx_popul in enumerate(populs):
            path_popul = path + f'/population_{indx_popul}'
            fitness_population = np.load(path_popul + f'/fitness_population_{indx_popul}.npy')
            ax.plot(fitness_population)
    if mode == 'delay':
        individ_names = [name for name in os.listdir(path) if os.path.isdir(path+'/'+name)]
        populations = [re.findall(r'population_\d+', name) for name in individ_names]
        populations = [individ[0] for individ in populations if individ != []]
        populations = np.sort(populations)
        for i_pop in range(len(populations)):
            pop_path = path + '/' + populations[i_pop]
            delay_names = [name for name in os.listdir(pop_path) if os.path.isdir(pop_path+'/'+name)]
            fitness_population = []
            for delay_name in delay_names:
                ind_path = pop_path+'/'+delay_name
                desired = np.load(ind_path + '/d.npy')
                noise_floor = np.load(ind_path + '/nf.npy')
                model_out = np.load(ind_path + '/y.npy')
                fitness_population.append(sl.nmse_nf(desired, desired - model_out, noise_floor))
            fitness_population.sort(reverse=True)
            ax.plot(np.array(fitness_population))
    return None

def animate_population(frame, populs, path, ax, y_lim=None):
    '''
    Function animates frame of the population located at the path 'path_popul'

    Parameters
    ----------
    frame : int
        Animation frame number.
    populs : list
        List of population indices.
    path : str
        Simulation folder path.
    ax : figure
        Figure class.
    ylim : list
        Y-axes limits

    Returns
    -------
    None.
    '''
    ax.clear()
    ax.set_xlabel('Individual index')
    ax.set_ylabel('NMSE [dB]')
    ax.set_title(f'Populations {populs[0]}-{populs[-1]}', fontsize=20)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True)
    popul = populs[frame]
    path_popul = path + f'/population_{popul}/fitness_population_{popul}.npy'
    fitness_population = np.load(path_popul)
    ax.plot(fitness_population)
    return None

def animate_evolution(populs, path, y_lim=[-12.5, -5], interval=300, repeat=False):
    '''
    Function animates evolution of the population fitness.
    Populations located at the simulation folder 'path'

    Parameters
    ----------
    populs : list
        List of population indices to take weights from.
    path : str
        Simulation folder path.
    ylim : list
        Y-axes limits.
    interval : int
        Interval between showing frames (milliseconds).
    repeat : bool
        Flag that shows whether to repear frames or not.

    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots()    
    ani = FuncAnimation(fig, partial(animate_population, populs=populs, path=path,
                  ax=ax, y_lim=y_lim), frames=len(populs), interval=interval, repeat=repeat, blit=False)
    return ani

def draw_weight_distr(populs, individs, path, trans2vec, name='H_init', 
                      mode='abs', nfig=None, ax=None, clf=True):
    '''
    Function draws the distributions of the weights which are located at the simulation
    path [path + '/population_i/individual_j/name']

    Parameters
    ----------
    populs : list
        List of population indices to take weights from.
    individs : list
        List of individuals indices to take weights from.
    path : str
        Simulation folder path.
    trans2vec : optional
        Link to the function that transforms loaded weights into the numpy array (vector)
    name : str
        Name of the weights distribution of which is supposed to be drawn. 
        The default is 'H_init' - internal layer weights of the MCVTDNN net.
    mode : str
        Flag that shows how to transform the weights before drawing:
        mode == 'abs' - weights = abs(weights)
        mode == 'real' - weights = real(weights)
        mode == 'imag' - weights = imag(weights)
    nfig : int
        Figure number. The default is None
    ax : figure
        Figure class. The default is None
    clf : bool
        Flag that shows whether to clear figure or not. The default is True

    Returns
    -------
    None.
    '''
    figsize_x = 7
    figsize_y = 5
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)  
    if clf:
        ax.cla()
    ax.set_xlabel('Initial vector elements')
    ax.set_ylabel('Magnitude')
    ax.set_title('Distributions', fontsize=20)
    ax.grid(True)
    if mode == 'abs':
        trans = np.abs
    if mode == 'real':
        trans = np.real
    if mode == 'imag':
        trans = np.imag
    for popul in populs:
        for individ in individs:
            individ_path = path + f'/population_{popul}/individual_{individ}/' + name
            weight = trans2vec(np.load(individ_path))
            p, x = np.histogram(trans(weight), bins=100)
            ax.plot(x[:-1], p)
            # sns_plot = sns.kdeplot(trans(weight), shade=False)
            # fig = sns_plot.get_figure()
            # ax.plot(trans(weight))
    return None

def del_folder_content(path : str):
    '''
    Function deletes content of the folder with the path 'path'

    Parameters
    ----------
    path : str
        Folder path.

    Returns
    -------
    None.
    '''
    for path in Path(path).glob('*'):
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()
    return None
         
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
	"""
	Deletes the attribute specified by the given list of names.
	For example, to delete the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'])
	"""
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		_del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
	"""
	This function removes all the Parameters from the model and
	return them as a tuple as well as their original attribute names.
	The weights must be re-loaded with `load_weights` before the model
	can be used again.
	Note that this function modifies the model in place and after this
	call, mod.parameters() will be empty.
	"""
	orig_params = tuple(mod.parameters())
	# Remove all the parameters in the model
	names = []
	for name, p in list(mod.named_parameters()):
		_del_nested_attr(mod, name.split("."))
		names.append(name)

	'''
		Make params regular Tensors instead of nn.Parameter
	'''
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
	"""
	Set the attribute specified by the given list of names to value.
	For example, to set the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'], value)
	"""
	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
	"""
	Reload a set of weights so that `mod` can be used again to perform a forward pass.
	Note that the `params` are regular Tensors (that can have history) and so are left
	as Tensors. This means that mod.parameters() will still be empty after this call.
	"""
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, x):
	'''

	@param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
	@param x: input since any gradients requires some input
	@return: either store jac directly in parameters or store them differently

	we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
	'''

	jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
	all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
	load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

	def param_as_input_func(model, x, param):
		load_weights(model, [name], [param]) # name is from the outer scope
		out = model(x)
		return out

	for i, (name, param) in enumerate(zip(all_names, all_params)):
		jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param, \
							 strict=True if i==0 else False, vectorize=False if i==0 else True)
		print(jac.shape, i)

	del jac_model # cleaning up
        
def check_command_file(path=None):
    '''
    Function checks 'command_file.txt' whether it includes command strings.
    If there is a skip-string, then function returns SKIP-flag.

    Parameters
    ----------
    path : str
        Path to the 'command_file.txt'. Default is None

    Returns
    -------
    Command flag : str.
    '''
    command_flag = None
    if path is not None:
        if path == '': 
            dash = '' 
        else: 
            dash = '/'
        try:
            command_file = open(path + dash + r'command_file.txt', 'r+')
            print('Waiting for your command...')
            text = command_file.read()
            # Process commands:
            if SKIP_FLAG in text:
                command_flag = SKIP_FLAG
            # if ...
            #   other commands ...
            if text != '' and text != 'done':
                command_file.truncate(0)
                command_file.seek(0, 0)
                command_file.write('done')
                command_file.close()
            return command_flag
        except: 
            pass
    else:
        return command_flag