# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:52:19 2021

@author: Saeid
"""

"""In this python script, the BinaryGA class is instantiated and then developed.
   The process of convergence and the values of best chromosomes are also plotted
   using the relevant functions defined in the BinaryGA class"""

# Importing the BinaryGA class from Utils   
from Utils.binary_ga import BinaryGA

# Instantiating the binary ga and implementing it.
ga = BinaryGA(n_var = 30, n_pop = 20, p_c = 0.8, p_m = 0.2)
NFE, BC, BP = ga.ga_compile(num_iter = 30,
                        min_max = "minimize",
                        mu = 0.3,
                        selection_type = 'roulette_wheel_selection', 
                        beta = 1,
                        tournament_size = 5,
                        cross_over_type = 'combined_crossover')

# Plotting the convergence procedure using plot_convergence function.
ga.plot_convergence(NFE, BC)

# Plotting a heatmap showing the values of the best chromosomes in each iteration of GA
ga.heatmap(NFE, BP)
