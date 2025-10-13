# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:30:16 2023

@author: dwx1065688
"""

import os
import sys
import shutil
import plot_lib as pl
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import feic_lib as feic
import feic_model_lib as feic_model
import support_lib as sl
import utils_nn as utils
import random
import copy

from deap import base, algorithms
from deap import creator
from deap import tools

'''
    Distribution functions
'''
randnint = lambda n : np.round(1.5*np.random.randn(n)).astype(int)
randnint_single = lambda : randnint(1)[0]

'''
    Create individual class
'''
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class GeneticDelay:
    '''
        Class for the search of optimal delays of any model
        by means of Genetic algorithm.
    '''
    def __init__(self, net, data, optimizer, init_optimizer : int, 
                 delay_bound : int, population_size : int, breed_num : int,
                 mutation_prob : float,
                 crossover_prob : float,
                 optimal_delays : list,
                 results_path = None):
        '''
        Parameters
        ----------
        net : optional
            Model class
        data : dict
            Dictionary data must include 3 keys:
            'input', value: numpy.ndarray with input signal.
                Here data['input'] is 1-dimensional array of complex signal (Not preprocessed).
            'target', value: numpy.ndarray with target signal
            'noise_floor', value: numpy.ndarray with noise floor signal
        delay_bound : int
            Delays would be searched in range [-delay_bound, delay_bound]
        optimizer : optional
            Optimizer class, with the current model optimization toolkit
        init_optimizer : optional
            Optimizer of initial weights. Could be MonteCarlo/Genetic.
        population_size : int
            Number of individuals in population.
        mutation_prob : float
            Each element (delay) of the individual probability
        optimal_delays : list
            Delays which are used in model and mustn`t be optimized.
            List optimal_delays is added to the whole list of delays just before the input signal preparing
        results_path : str
            Simulation directory path. Default is None

        Returns
        -------
        None.
        '''
        self.net = net
        self.data = copy.deepcopy(data)
        self.optimizer = optimizer
        self.init_optimizer = init_optimizer
        self.population_size = population_size
        self.breed_num = breed_num
        self.delay_bound = delay_bound
        self.optimal_delays = optimal_delays
        assert len(optimal_delays) < net.delays_num*net.nonlin_num, \
            f'Optimal delays list length {len(optimal_delays)} must be strictly lower than whole number of delays in model {net.delays_num*net.nonlin_num}'
        self.individual_length = net.delays_num*net.nonlin_num - len(optimal_delays)
        self.results_path = results_path
        if self.results_path is not None:
            self.save_results = True
        else:
            self.save_results = False
        # Mutation properties
        self.mutation_prob = mutation_prob
        # Crossover properties
        self.crossover_prob = crossover_prob
        # Define population class
        self.toolbox = base.Toolbox()
        self.toolbox.register("generate_delay", randnint_single)
        self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual, self.toolbox.generate_delay, self.individual_length)
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.individualCreator)
        # Create population
        self.population = self.toolbox.populationCreator(n=self.population_size)
        # Register genetic algorithm functions
        self.toolbox.register("evaluate", self.calc_fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxUniform, indpb=self.crossover_prob)
        self.toolbox.register("mutate", tools.mutGaussian, indpb=self.mutation_prob, mu=0, sigma=0.5)
        # Constraint: at the end of population modification delays must be integer
        self.toolbox.decorate("mutate", self.checkBounds())
        # Define statistics class
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min)
        self.stats.register("avg", np.mean)
        self.logbook = tools.Logbook()
        # Auxiliary objects
        self.curr_individual = 0
        self.curr_population = 0
        # List of cheked delays
        self.delays_list = []
        self.delays_fitnesses = []
        return None
    
    def calc_fitness(self, individual):
        '''
        Parameters
        ----------
        individual : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        '''
        if self.save_results == True:
            try:
                best_delays_path = self.results_path + r'/best_delays'
                best_delays_path += r'/' + os.listdir(best_delays_path)[0]
                self.cost_best = self.optimizer.load_cost(path=best_delays_path)
            except:
                try: 
                    os.mkdir(path=self.results_path + r'/best_delays')
                except:
                    pass
                self.cost_best = 100
        self.curr_population = self.curr_individual // self.population_size
        # Create population folder if there is no such individuals
        if not (individual in self.delays_list):
            individual_path = self.create_folder(individual)
            # Point the individual class
            self.init_optimizer.results_path = individual_path
            # Prepare input signal
            delays = self.optimal_delays + individual
            delays = np.array_split(delays, self.net.nonlin_num)
            delays = [delay.tolist() for delay in delays]
            input_data = utils.dataset_prepare(self.data['input'].reshape((-1)), delays)
            self.optimizer.load_input_data(copy.deepcopy(input_data))
            self.init_optimizer.load_input_data(copy.deepcopy(input_data))
            self.init_optimizer.train()
            fitness = self.init_optimizer.cost_best
            # Save weights and arrays of the best individual
            if fitness < self.cost_best:
                self.cost_best = fitness
                # Delete 'best_delays' folder
                shutil.rmtree(path=self.results_path + r'/best_delays')
                # Create 'best_delays' and copy best individual to the 'best_delays'
                shutil.copytree(src=individual_path, dst=self.results_path + r'/best_delays/' + str(individual))
            # Add new fitnesses and individuals to the memory list
            self.delays_fitnesses.append(fitness)
            self.delays_list.append(individual)
        else:
            fitness = self.delays_list.index(individual)
        print(f'Genetis delays current best NMSE: {self.cost_best} dB')
        self.curr_individual += 1
        return fitness,
    
    def create_folder(self, individual):
        '''
        Function generates folders in simulation folder, which 
        correspond to different delays sets.

        Returns
        -------
        Current individual path
        '''
        popul_dir_name = f'population_{self.curr_population}'
        try:
            os.mkdir(self.results_path+'/'+popul_dir_name)
        except:
            pass
        # Create individual folder
        individual_path = self.results_path+'/'+popul_dir_name+f'/{str(individual)}'
        try:
            os.mkdir(individual_path)
        except OSError as error:
            print(error)
        return individual_path
    
    def visualize(self):
        '''
        Function which visualizes genetic algorithm results.

        Returns
        -------
        None.
        '''
        minFitnessValues, meanFitnessValues = self.logbook.select("min", "avg")
        plt.plot(minFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Поколение')
        plt.ylabel('Макс/средняя приспособленность')
        plt.title('Зависимость максимальной и средней приспособленности от поколения')
        plt.show()
        return None
    
    def checkBounds(self):
        '''
        Function keeps delays within the predefined bounds: [-delay_bound, delay_bound].
        
        Returns
        -------
        Decorator.
        '''
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        child[i] = int(np.round(child[i]))
                        sign = np.sign(child[i])
                        child_abs = np.abs(child[i])
                        child[i] = sign * (child_abs % self.delay_bound)
                return offspring
            return wrapper
        return decorator
    
    def train(self):
        '''
        Optimal delays search by means of genetic algorithm
        
        Returns
        -------
        None.
        '''
        self.population, self.logbook = algorithms.eaSimple(self.population, self.toolbox,
                                        cxpb=self.crossover_prob,
                                        mutpb=self.mutation_prob,
                                        ngen=self.breed_num,
                                        stats=self.stats,
                                        verbose=True)
        # self.visualize()
        return None