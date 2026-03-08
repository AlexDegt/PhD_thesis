# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:50:33 2023

@author: dWX1065688
"""

import os
import sys
import numpy
import numpy as np
from time import perf_counter  
import copy
import support_lib as sl
import train as tr

class InitCondOpt:
    '''
        Class with the toolkit for initial conditions optimizations.
        Current class is a parent for optimizations such as Monte Carlo, Genetic etc
    '''
    # РЕАЛИЗОВАТЬ ИДЕЮ С Init constant layers в set_init_weights!!!!!
    def __init__(self, net, data, optimizer, cost_best=0, results_path=None, curr_results_print=True, threshold=-0.5):
        '''
        Constructor
        
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
        cost_best : float
            Best cost function value to compare with
        results_path : str
            Path to save results of the simulations. Default is None.
            If results_path == None - results will not be saved

        Returns
        -------
        None.
        '''
        self.net = net
        self.data = data
        self.optimizer = optimizer
        self.x = self.data['input']
        self.d = self.data['target']
        self.nf = self.data['noise_floor']
        self.cost_best = cost_best
        self.results_path = results_path
        self.curr_results_print = curr_results_print
        self.threshold = threshold
        if type(self.results_path) == str:
            self.save_results = True
        else:
            self.save_results = False
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
        self.data['input'] = input_data
        self.x = input_data
        return None
    
class MonteCarlo(InitCondOpt):
    '''
        The class which represents Monte Carlo initial
        condition optimization
    '''
    def __init__(self, net, data, optimizer, start_num=30, cost_best=0, results_path=None, curr_results_print=True, threshold=-0.5):
        '''
        Constructor
        
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
        start_num : int
            Number of starts with different initial conditions, chosen by 
            Monte Carlo method. Default is 30
        cost_best : float
            Best cost function value to compare with
        results_path : str
            Path to save results of the simulations. Default is None.
            If results_path == None - results will not be saved

        Returns
        -------
        None.
        '''
        InitCondOpt.__init__(self, net, data, optimizer, cost_best, results_path, curr_results_print, threshold)
        self.start_num = start_num
        return None
    
    def train(self, init_weights={}):
        '''
        Function trains self.net by self.optimizer means on self.data.
        Function trains it self.start_num times and choses initial
        coefficients with the best cost function value

        Parameters
        ----------
        init_weights : dict
            Training initial weights. The default is {}.

        Returns
        -------
        None.
        '''
        if self.save_results == True:
            try:
                self.net.set_weights(self.net.load_weights(path=self.results_path, add_info=''))
                y = self.net.forward(self.x)
                self.cost_best = sl.nmse_nf(self.d, self.d-y, self.nf)
            except:
                self.cost_best = 0
        if self.curr_results_print == True:
            print(f'Current best NMSE: {self.cost_best} dB')
        interrupt_count = 0
        for i in range(self.start_num):
            t_start = perf_counter()
            if init_weights == {}:
                self.net.set_init_weights()
            else:
                self.net.set_weights(init_weights, expand_dim=False)
            init_weight = copy.deepcopy(self.net.weight)
            tr.train(self.net, self.data, self.optimizer, results_path=self.results_path)
            y = self.net.forward(self.x)
            NMSE = sl.nmse_nf(self.d, self.d-y, self.nf)
            if NMSE < self.cost_best:
                self.cost_best = NMSE
                if self.save_results == True:
                    self.net.save_weights(path=self.results_path)
                    self.net.save_weights(path=self.results_path, weights=init_weight, add_info='_init')
                    np.save(self.results_path+'/step_evolution.npy', self.optimizer.get_step_evol())
                    np.save(self.results_path+'/loss_evolution.npy', self.optimizer.get_loss_evol())
                    np.save(self.results_path+'/d.npy', self.d)
                    np.save(self.results_path+'/y.npy', y)
                    np.save(self.results_path+'/nf.npy', self.nf)
            t_end = perf_counter()
            print(f'{i}. NMSE = {NMSE} dB, time elapsed: {t_end-t_start} s')
            if NMSE >= self.threshold:
                interrupt_count += 1
            if interrupt_count == 3:
                break
        print(f'Monte Carlo best NMSE: {self.cost_best} dB')
        return None

class Individual(InitCondOpt):
    '''
        Class of individuals in generation of genetic algorithm
    '''
    def __init__(self, net, weight=None):
        '''
        Constructor

        Parameters
        ----------
        weight : dict
            Initial weights supposed to be optimized. The default is None.
            Individual weights are already vectorized.

        Returns
        -------
        None.
        '''
        self.net = copy.copy(net)
        # Better to make deepcopy here for the case when more parameters would
        # be optimized (for example b_ext of TDNN).
        # Change of vaiable (not array or dict) don`t change its link,
        # that`s why it must be copied for the case it would be optimized
        self.net.weight = copy.deepcopy(net.weight)
        if weight == None:
            # Initial weights are set randomly if weight == None
            self.net.set_init_weights()
        else:
            self.net.set_weights(weight)
        # Non-vectorized initial weights
        self.init_weight = copy.deepcopy(self.net.weight)
        weight = self.net.init2vec()
        self.set_weights(weight)
        self.fitness = 0
        self.step_evolution = []
        self.fitness_evolution = []
        return None
    
    def set_weights(self, weight):
        '''
        Sets vectorized coefficients weights to the network.
        Weights must have dimensions: nonlin_num*K*(delays_num - 1)

        Returns
        -------
        None.
        '''
        self.net.vec2init(weight)
        self.weight = self.net.init2vec()
        return None
    
    def get_weights(self):
        '''
        Returns
        -------
        Link to the vectorized weights of the individual.
        '''
        return self.weight
    
    def get_step_evol(self):
        '''
        Returns
        -------
        Learning rate/adaptation step evolution curve.
        '''
        return np.array(self.step_evolution)
    
    def get_fitness_evol(self):
        '''
        Returns
        -------
        Fitness evolution during adaptation.
        '''
        return np.array(self.fitness_evolution)
        

class Genetic(InitCondOpt):
    '''
        The class which represents initial condition optimization
        by the Genetic algorithm
    '''
    def __init__(self, net, data, optimizer, population_size=100, select_ratio=0.9, \
                 crossover_shift=2, crossover_prob=0.9, crossover_type='half_shift&help_cross2middle', \
                 mutation_prob=0.15, mutation_sigma=0.5, mutation_type='normal_real_int', \
                 breed_num=30, cost_best=0, results_path=None):
        '''
        Constructor
        
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
        select_ratio : float
            Ratio of selected individuls to whole individuals number
            in generation
        crossover_shift : int
            Parameter that represents the shift value of best half of the population.
            Best half of population is shifted accrding to current value,
            after that for the shifted and non-shifted arrays crossover is implemented
        crossover_prob : float
            Probability of the crossover cell activation. Crossover cell means crossover
            between several individuals (NOT the whole population)
        crossover_type : str
            Type of the crossover function to be used in genetic learning.
        mutation_prob : float
            Probability of the mutation cell activation. Mutation cell means 
            mutation of the one individual (NOT the whole population)
        mutation_sigma : float
            Gain coefficient of the normal distribution, which is the source 
            of the mutation vector in mutation cell
        breed_num : int
            Number of generations of genetic algorithm (start_num equivalent).
            Default is 30
        cost_best : float
            Best cost function value to compare with
        results_path : str
            Path to save results of the simulations. Default is None.
            If results_path == None - results will not be saved

        Returns
        -------
        None.
        '''
        if crossover_type == 'half_rand&help_cross3rhomb':
            assert population_size >= 6, \
                f'Population size ({population_size}) must be higher, than 6 in case of using 3-individual crossover.'
        InitCondOpt.__init__(self, net, data, optimizer, cost_best, results_path)
        # Gentic algorithm attributes
        self.population_size = population_size
        self.breed_num = breed_num
        # Selection attributes
        self.select_ratio = select_ratio
        # Crossover attributes
        self.crossover_shift = crossover_shift
        self.crossover_prob = crossover_prob
        self.crossover_type = crossover_type
        # Mutation attributes
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.mutation_type = mutation_type
        # Creates list self.population
        self.population = self.create_population(populat_size=self.population_size)
        self.populat_fitness = np.zeros((self.population_size,), dtype=float)
        # Number of current population
        self.curr_population = 0
        # Time elapsed for the one population
        self.time_popul = 0
        return None
    
    def calc_fitness(self, Individual):
        '''
        Calculates fitness of the appropriate Individual 
        on base its vectorized model weights

        Returns
        -------
        Fitness.
        '''
        Individual.net.weight = copy.deepcopy(Individual.init_weight)
        Individual.net.vec2init(weight_vec=Individual.weight)
        Individual.init_weight = copy.deepcopy(Individual.net.weight)
        tr.train(Individual.net, self.data, self.optimizer, results_path=self.results_path)
        Individual.step_evolution = copy.copy(self.optimizer.step_evoluiton)
        Individual.fitness_evolution = copy.copy(self.optimizer.loss_evolution)
        y = Individual.net.forward(self.x)
        Individual.fitness = sl.nmse_nf(self.d, self.d - y, self.nf)
        return Individual.fitness
    
    def create_population(self, populat_size: int):
        '''
        Function creates population of Invdividuals and put
        them into the list self.population

        Parameters
        ----------
        populat_size : int
            Size of population to be created.

        Returns
        -------
        None.
        '''
        population = [Individual(self.net) for i_individ in range(populat_size)]
        return population
    
    def population_fitness(self, population):
        '''
        Function calculates fitness for each of the individuals in
        population and write this results to the populat_fitness list
        
        Parameters
        ----------
        population : class Individual
            Population individuals fitness of which 
            supposed to be calculated.
        
        Returns
        -------
        List of fitnesses of each individual in the population.
        '''
        # populat_fitness = list(map(self.calc_fitness, population))
        populat_fitness = []
        population_size = len(population)
        for i_individ in range(population_size):
            tmp = self.calc_fitness(population[i_individ]).item()
            populat_fitness.append(tmp)
            print(f'Individual {i_individ+1}/{population_size} is fit.')
        return populat_fitness
    
    def selection(self):
        '''
        Function sorts population individuals in the order
        of fitness improvement

        Returns
        -------
        None.
        '''
        # Create new fresh random individuals and add them to the whole population
        ind_save = numpy.ceil(self.population_size*(1 - self.select_ratio)).astype(int)
        fresh_individs = self.create_population(populat_size=ind_save)
        fresh_individs.extend(self.population)
        self.population = fresh_individs
        self.populat_fitness  = self.population_fitness(population=self.population)
        # Sort individuals in population according to their fitness
        arx = zip(self.population, self.populat_fitness)
        arx_sorted = sorted(arx, key=lambda x: x[1])
        arx_sorted = list(map(list, arx_sorted))
        self.population = [arx_sorted[i][0] for i in range(len(arx_sorted))]
        self.populat_fitness = [arx_sorted[i][1] for i in range(len(arx_sorted))]
        self.population = self.population[ind_save:]
        self.populat_fitness = self.populat_fitness[ind_save:]
        return None
    
    def crossover_2middle_cell(self, individ_0, individ_1):
        '''
        Crossover of 2 individuals.
        Child is the middle point between points related to
        0-th and 1-st individuals.
        
        Parameters
        ----------
        individ_0, individ_1 : class Individual
            Individuals which are exploited to
            calculate crossover weights.
        
        Returns
        -------
        None.
        '''
        child = copy.deepcopy(individ_0)
        if numpy.random.random() < self.crossover_prob:
            child.weight = (individ_0.weight + individ_1.weight)/2
        else:
            child.weight = individ_0.weight
        return child
    
    def crossover_3rhombus_cell(self, individ_0, individ_1, individ_2):
        '''
        Crossover of 3 individuals.
        Child is the point built up to the rhombus vertex,
        such that rhombus has vertices related to individuals:
            0-th, 1-st, 2-nd and the child.
        
        Parameters
        ----------
        individ_0, individ_1, individ_2: class Individual
            Individuals which are exploited to
            calculate crossover weights.
        
        Returns
        -------
        None.
        '''
        child = copy.deepcopy(individ_0)
        if numpy.random.random() < self.crossover_prob:
            child.weight = individ_1.weight + individ_2.weight - individ_0.weight
        else:
            child.weight = individ_0.weight
        return child
    
    def crossover(self):
        '''
        Function that implements crossover between individuals
        in the population with appropriate probability

        Returns
        -------
        None.
        '''
        if self.crossover_type == 'half_shift&help_cross2middle': 
            ind_middle = int(np.floor(self.population_size/2))
            popul_worse_part = self.population[:ind_middle]
            popul_best_part = self.population[ind_middle:]
            popul_best_part_inv = popul_best_part[::-1]
            popul_best_part_shift = np.roll(popul_best_part, self.crossover_shift)
            len_worse_part = min(len(popul_worse_part), len(popul_best_part_inv))
            len_best_part = len(popul_best_part)
            crossed_worse_part = [self.crossover_2middle_cell(popul_worse_part[i], popul_best_part_inv[i]) for i in range(len_worse_part)]
            crossed_best_part = popul_best_part
            # crossed_best_part = [self.crossover_cell_half_shift_help(popul_best_part[i], popul_best_part_shift[i]) for i in range(len_best_part)]
            crossed_worse_part.extend(crossed_best_part)
            self.population = crossed_worse_part
        if self.crossover_type == 'half_rand&help_cross3rhomb':
            ind_middle = int(np.floor(self.population_size/2))
            popul_worse_part = self.population[:ind_middle]
            popul_best_part = self.population[ind_middle:]
            len_worse_part = min(len(popul_worse_part), len(popul_best_part))
            len_best_part = len(popul_best_part)
            crossed_worse_part = []
            crossed_best_part = []
            for i in range(len_worse_part):
                rand_best_0 = numpy.random.randint(len_worse_part)
                rand_best_1 = numpy.random.randint(len_worse_part)
                while rand_best_1 == rand_best_0:
                    rand_best_1 = numpy.random.randint(len_worse_part)
                child = self.crossover_3rhombus_cell(popul_worse_part[i], popul_best_part[rand_best_0], popul_best_part[rand_best_1])
                crossed_worse_part.append(child)
            for i in range(len_best_part):
                rand_best_0 = numpy.random.randint(len_best_part)
                while rand_best_0 == i:
                    rand_best_0 = numpy.random.randint(len_best_part)
                rand_best_1 = numpy.random.randint(len_best_part)
                while rand_best_1 == rand_best_0 or rand_best_1 == i:
                    rand_best_1 = numpy.random.randint(len_best_part)
                child = self.crossover_3rhombus_cell(popul_best_part[i], popul_best_part[rand_best_0], popul_best_part[rand_best_1])
                crossed_best_part.append(child)
            crossed_worse_part.extend(crossed_best_part)
            self.population = crossed_worse_part
        return None
    
    def mutation_cell(self, individ):
        '''
        Function calculates mutation for the input individual
        
        Parameters
        ----------
        individ : class Individual
            Individual which is supposed to be mutated.
        
        Returns
        -------
        None.
        '''
        axis_0_shape = individ.weight.shape[0]
        mask = []
        for i in range(axis_0_shape):
            if numpy.random.random() < self.mutation_prob:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask)
        if self.mutation_type == 'normal_complex_float':
            postproc = lambda x : x
            mutation = np.random.randn(axis_0_shape,) + 1j*np.random.randn(axis_0_shape,)
        if self.mutation_type == 'normal_real_int':
            postproc = np.round
            mutation = np.random.randn(axis_0_shape,)
        mutation = postproc(self.mutation_sigma*mutation) * mask
        individ.weight += mutation
        individ.weight = (individ.weight.real % self.net.get_init_lim()).astype(complex)
        return individ
    
    def mutation(self):
        '''
        Function implements mutation for all individuals in 
        population with appropriate propability

        Returns
        -------
        None.
        '''
        self.population = [self.mutation_cell(self.population[i]) for i in range(self.population_size)]
        return None
    
    def load_population(self, individs, path, popul_num):
        '''
        Function loads individuals from definite population.
        Function loads individuals` weights, initial weights; sets fitnesses for
        each indiviual of population, step evolution, fitness evolution, population fitness.

        Parameters
        ----------
        individs : list
            List of individuals indices to be loaded.
        path : str
            Path to the simulation folder to load weights from.
        popul_num : int
            Index of the population to load individuals` weigths.

        Returns
        -------
        None.
        '''
        population_path = path + f'/population_{popul_num}'
        len_individs = len(individs)
        assert self.population_size >= len_individs, \
            f'Current population size {self.population_size} must be higher, than number of individuals to load {len_individs}'
        for j, indx_individ in enumerate(individs):
            individ_path = population_path + f'/individual_{indx_individ}'
            # Load optimal weights
            weight_opt = self.population[j].net.load_weights(path=individ_path, add_info='')
            self.population[j].net.set_weights(weight_opt)
            # Load initial weights
            self.population[j].init_weight = self.population[j].net.load_weights(path=individ_path, add_info='_init')
            # Set vectorized weights:
            self.population[j].set_weights(self.population[j].net.init2vec(self.population[j].init_weight))
            # Load step evolution
            self.population[j].step_evolution = np.load(individ_path + '/step_evolution.npy')
            # Load loss evolution
            self.population[j].loss_evolution = np.load(individ_path + '/fitness_evolution.npy')
            # Load signal arrays to calculate individuals fitness
            input_data = np.load(individ_path + '/x.npy')
            target_data = np.load(individ_path + '/d.npy')
            noise_floor_data = np.load(individ_path + '/nf.npy')
            # Calculate and set individuals` fitness
            model_output = self.population[j].net.forward(input_data)
            error = target_data - model_output
            self.population[j].fitness = sl.nmse_nf(target_data, error, noise_floor_data).item()
            self.populat_fitness[j] = self.population[j].fitness
        self.crossover()
        self.mutation()
        return None
    
    def save_individ(self, individ, path):
        '''
        Function saves whole train progress:
            1) Optimal weights of the best individual
            2) Initial weights of the best individual
            3) Model output with weights of the best individual
            4) Interference array
            5) Noise floor array
            6) Each population fitness

        Parameters
        ----------
        individ : class Individual
            Individual, which info supposed to be saved.
        path : str
            Path to save info about current individual.
    
        Returns
        -------
        None.
        '''
        # Save optimal weights of the best individual
        individ.net.save_weights(path=path, add_info='')
        # Obtain and save suppressed interference
        y = individ.net.forward(self.x)
        np.save(path+'/x.npy', self.x)
        np.save(path+'/d.npy', self.d)
        np.save(path+'/y.npy', y)
        np.save(path+'/nf.npy', self.nf)
        # Obtain initial weights of the current individual
        # individ.net.weight = copy.deepcopy(individ.init_weight)
        # Save initial weights of the best individual
        individ.net.save_weights(path=path, weights=individ.init_weight, add_info='_init')
        # Save learning curve and step evolution of the best individual
        np.save(path+'/step_evolution.npy', individ.get_step_evol())
        np.save(path+'/fitness_evolution.npy', individ.get_fitness_evol())
        return None
    
    def log_progress(self):
        '''
        Function saves whole train progress:
            1) Optimal weights of the best individual/whole population
            2) Initial weights of the best individual/whole population
            3) Model output with weights of the best individual/whole population
            4) Interference array
            5) Noise floor array
            6) Each population fitness

        Returns
        -------
        None.
        '''
        # Best NMSE from current population
        NMSE = min(self.populat_fitness)
        init_cost_best = self.cost_best
        if NMSE < self.cost_best:
            self.cost_best = NMSE
        # Save optimal weights of the best individual
        try:
            os.mkdir(self.results_path+f'/population_{self.curr_population}')
        except OSError as error:
            print(error)
        try:
            os.mkdir(self.results_path+'/best_individual')
        except OSError:
            pass #print(error)  
        if self.save_results == True and NMSE < init_cost_best:
            ind_best = numpy.argmin(self.populat_fitness)
            individ_best = self.population[ind_best]
            self.save_individ(individ_best, path=self.results_path+'/best_individual')
        for i_individ in range(len(self.population)):
            try:
                os.mkdir(self.results_path+f'/population_{self.curr_population}/individual_{i_individ}')
            except OSError as error:
                print(error)
            self.save_individ(self.population[i_individ], path=self.results_path+f'/population_{self.curr_population}/individual_{i_individ}')
        # Save current population fitness
        np.save(self.results_path+f'/population_{self.curr_population}/fitness_population_{self.curr_population}', self.populat_fitness)
        print(f'Breed: {self.curr_population}. NMSE = {NMSE} dB, time elapsed: {self.time_popul} s')
        return None
        
    def train(self):
        '''
        Function trains self.net by self.optimizer means on self.data.
        Function trains it self.breed_num times and choses initial
        coefficients with the best cost function value

        Returns
        -------
        None.
        '''
        if self.save_results == True:
            try:
                self.net.set_weights(self.net.load_weights(path=self.results_path+r'/best_individual', add_info=''))
                y = self.net.forward(self.x)
                self.cost_best = sl.nmse_nf(self.d, self.d-y, self.nf)
            except:
                self.cost_best = 0
        print(f'Current best NMSE: {self.cost_best} dB')
        for curr_breed in range(self.breed_num):
            t_start = perf_counter()
            self.curr_population = curr_breed
            self.selection()
            self.crossover()
            self.mutation()
            t_end = perf_counter()
            self.time_popul = t_end - t_start
            self.log_progress()
        print(f'Best NMSE: {self.cost_best} dB!')
        return None