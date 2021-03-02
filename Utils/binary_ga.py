# -*- coding: utf-8 -*-

"""
Created on Wed Jun  3 12:07:29 2020

@author: Saeid
"""
"""
This is a binary Genetic algorithm which minimizes the summation of a vector of size (n_var) containing arrays of 0 and 1
only. To this end, a population of chromosomes (n_pop) are chosen randomly at the beginning of the process. Fractions
of this population are chosen for the processes of crossover and mutation, denoted by p_c and p_m, respectively.
Subsequently, a number of iterations for conducting the crossover and mutation is chosen (num_iter) in the ga_compile
function. Different selection types such as roulette wheel, tournament and random selection along with various 
types of crossover namely uniform, single-point and double-point crossover can be utilized. Finally, the converegance of
the process in finding the minimum amount of vectors.
"""

# Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
np.random.seed(42)

class BinaryGA:
    
    def __init__(self, n_var, n_pop, p_c, p_m):
        
        self.n_var = n_var
        self.n_pop = n_pop
        self.p_c = p_c
        self.p_m = p_m
    
    # In this function, the cost which represents the summation of the binary vector of size n_var is created.
    # Number of Function Evaluation (NFE) is also defined as a global variable through the class functions
    def cost_function(self, x):
        global NFE
        try:
            NFE +=1
        except:
            NFE = 0
        z = sum(x)
        return z
    
    # A function for building a dictionary containing a new set of chromosomes
    def new_chrom(self):
        return {'Position':[],
                    'Cost':[]}  
        
    # In the following four functions, various types of crossover functions are created.
    ## Single Point Crossover
    def single_point_crossover(self, x1, x2):
        nVar = np.size(x1)
        c = np.random.randint(1,nVar)
        y1 = np.concatenate([x1[0:c], x2[c:]], axis = 0)
        y2 = np.concatenate([x2[0:c], x1[c:]], axis = 0)
        return y1, y2
    
    ## Double Point Crossover
    def double_point_crossover(self, x1, x2):
        nVar = np.size(x1)
        c = np.random.randint(1,nVar,2)
        c1 = min(c)
        c2 = max(c)
        y1 = np.concatenate([x1[0:c1], x2[c1:c2], x1[c2:]], axis = 0)
        y2 = np.concatenate([x2[0:c1], x1[c1:c2], x2[c2:]], axis = 0)
        return y1, y2
    
    ## Uniform Crossover
    def uniform_crossover(self, x1, x2):
        nVar = np.size(x1)
        alpha = np.random.randint(0,2,nVar)
        y1 = alpha*x1+(1-alpha)*x2
        y2 = alpha*x2+(1-alpha)*x1
        return y1, y2
    
    ## The combination of three random models for Crossover
    def crossover(self, x1, x2):
        METHOD = np.random.randint(1,4)
        if METHOD == 1:
            y1, y2 = self.single_point_crossover(x1,x2)
        elif METHOD == 2:
            y1, y2 = self.double_point_crossover(x1,x2)
        else:
            y1, y2 = self.uniform_crossover(x1,x2)
        return y1, y2
    
    # In the following two functions, different types of gene selection are defined.
    ## Roulette Wheel Selection
    def roulette_wheel_selection(self, prob):
        r = np.random.rand()
        c = np.cumsum(prob)
        i = min(np.where(r < c))
        return i[0]
    
    ## Tournament Selection
    def tournament_selection(self, pop, tournament_size):
        n_pop = len(pop['Position'])
        # Selecting m Parents Randomly (m = TournamentSize)
        S = np.random.randint(0, n_pop, tournament_size) 
        # Finding the Minimum Cost in the Seleced Population    
        T = self.new_chrom()
        for i in S:
            T['Position'].append(pop['Position'][i])
            T['Cost'].append(pop['Cost'][i])
        T = self.sortieren(T)
        
        # Finding an index in Population with the Minimum Cost
        for i in range(n_pop):
            if pop['Cost'][i] == T['Cost'][0]:
                j = i
                break
        return j        
    
    # In this funcion, the mutatuion process is defined.
    def mutate(self, x, mu):
        nVar = len(x)
        NE = int(round(mu*nVar))     # Number of elements chosen for mutation
        j = np.random.randint(0,nVar,NE)
        y = x.copy()
        for i in j:
            y[j] = 1-x[j]
        return y
    
    # In this function, all of the created Chromosomes through mutation and crossover along with the parents are merged.
    def merge(self, pop, popC, popM):
        pop_merged = self.new_chrom()
        pop_merged['Position'] = pop['Position']+popC['Position']+popM['Position']
        pop_merged['Cost'] = pop['Cost']+popC['Cost']+popM['Cost']
        return pop_merged
    
    # In this function, a set of chromosomes is sorted ascending or descending.
    def sortieren(self, pop, inverse = False):
        """Generating 2 temporary Populations in order to 
           make tuples with positions and costs of every Population"""
        pop_cost = pop['Cost']
        pop_position = pop['Position']

        # Sorting
        sorted_index  = sorted(range(len(pop_cost)), key=lambda k: pop_cost[k])
        pop_cost_sorted = [pop_cost[k] for k in sorted_index]
        pop_position_sorted = [pop_position[k] for k in sorted_index]
        
        # Generating the Sorted Population
        pop_sorted = self.new_chrom()
        
        for i in range (len(pop['Position'])):
            
            pop_sorted['Cost'].append(pop_cost_sorted[i])
            pop_sorted['Position'].append(pop_position_sorted[i])
        
        if inverse == True:
            pop_sorted['Cost'].reverse()
            pop_sorted['Position'].reverse()
            
        return pop_sorted
    
    # In this function the process of GA is initialized and a set of chromosomes with the size of n_pop is created.
    def ga_init(self):
    
        # Number of Function Evaluation Counter
        global NFE
        NFE = 0
        
        pop = self.new_chrom()
        var_size = [1, self.n_var]
        for i in range(self.n_pop):
        
            # Initialize Position:
            Position = np.random.randint(0,2,var_size)[0]
            Cost = self.cost_function(Position)
            
            pop['Position'].append(Position)
            pop['Cost'].append(Cost)
        
        return pop
    
    # In this function the process of crossover is generated.
    # The defined crossover functions, namely "single_point_crossover", "double_point_crossover", "uniform_crossover",
    # "crossover", and the defined selection functions namely, "roulette_wheel_selection" and "tournament_selection" are used.
    # The output of this function is a set of chromosomes with the size of n_pop*p_c
    def ga_crossover(self, pop, selection_type, beta = 0.1, tournament_size = 5, cross_over_type = 'uniform_crossover'):
        
        # Selecting Probability Pressure
        P = []
        
        # After some iterations all the costs are equal to 0 and Worstcost is 0
        pop['Cost']
        WorstCost = max(pop['Cost'])
        if WorstCost == 0:
            WorstCost = 1
            
        for i in range(self.n_pop):
            PP = np.exp(-beta*pop['Cost'][i]/WorstCost)
            P.append(PP)
        P = P/sum(P)
        
        # Preparing the First Parent
        popC1 = self.new_chrom()
        
        # Preparing the Second Parent
        popC2 = self.new_chrom()
        
        n_c = len(pop['Position'])*self.p_c
        for k in range(int(n_c/2)):
            
            # Select Parents Indices
            if selection_type == 'random_selection':
                i1 = np.random.randint(0,self.n_pop)
                i2 = np.random.randint(0,self.n_pop)
            
            elif selection_type == 'roulette_wheel_selection':
                i1 = self.roulette_wheel_selection(P)
                i2 = self.roulette_wheel_selection(P)
            
            elif selection_type == 'tournament_selection':
                i1 = self.tournament_selection(pop,tournament_size)
                i2 = self.tournament_selection(pop,tournament_size)
            
            # Select Parents
            Position1 = pop['Position'][i1]
            Position2 = pop['Position'][i2]
            
            # Apply Crossover
            if cross_over_type == 'uniform_crossover':
                PositionC1, PositionC2 = self.uniform_crossover(Position1,Position2)
            
            elif cross_over_type == 'single_point_crossover':
                PositionC1, PositionC2 = self.single_point_crossover(Position1,Position2)
            
            elif cross_over_type == 'double_point_crossover':
                PositionC1, PositionC2 = self.double_point_crossover(Position1,Position2)
            
            elif cross_over_type == 'combined_crossover':
                PositionC1, PositionC2 = self.crossover(Position1,Position2)
            
            CostC1 = self.cost_function(PositionC1)
            CostC2 = self.cost_function(PositionC2)
            # Assembling the Offsprings
            popC1['Position'].append(PositionC1)
            popC2['Position'].append(PositionC2)
            popC1['Cost'].append(CostC1)
            popC2['Cost'].append(CostC2)
            
        # Assembling the Offsprings from Parent 1 and Parent 2
        popC = self.new_chrom()
        popC['Position'] = popC1['Position'] + popC2['Position']
        popC['Cost'] = popC1['Cost'] + popC2['Cost']
        
        return popC

    # In this function the process of mutation is generated.
    # The defined mutation function, namely mutate is used.
    # The output of this function is a set of chromosomes with the size of n_pop*p_c    
    def ga_mutation(self, pop, mu):
        
        popM = self.new_chrom()
        
        n_m = np.ceil(len(pop['Position'])*self.p_m)
        for k in range(int(n_m)):
            
            # Select a random Parents Indice
            i = np.random.randint(0,self.n_pop)
            
            # Select Parent 
            Position = pop['Position'][i]
            
            # Apply Mutation on the selected Parent
            PositionM = self.mutate(Position, mu)
            
            # Evaluate the Cost of the Mutant of the selected Parent
            CostM = self.cost_function(PositionM)
    
            # Apply Mutation and Generate Mutants
            popM['Position'].append(PositionM)
            popM['Cost'].append(CostM)
            
        return popM
    
    # In this function, the whole process of GA is generated.
    # The outputs of this function are the Number of Function Evaluation(NFE), best position and best cost at the end of each iteration.
    def ga_compile(self, num_iter, mu, min_max = 'minimize', selection_type = 'roulette_wheel_selection', beta = 0.01, tournament_size = 20, cross_over_type = 'uniform_crossover'):
    
        # Collecting the Initializing Population:
        pop = self.ga_init()
        
        # Array to Hold Best Cost Values
        BestCost = []
        BestPosition = []
        
        # Array to Hold Number of Function Evaluation
        nfe = np.zeros(num_iter)
        
        # Main Loop
        for it in range(num_iter):
            
            # Crossover
            popC = self.ga_crossover(pop, selection_type, beta, tournament_size, cross_over_type)
        
            # Mutation
            popM = self.ga_mutation(pop, mu)
                
            # Creat Merged Population   
            pop_merged = self.merge(pop, popC, popM)
            
            # Sort Merged Population    
            if min_max == 'minimize':
                pop_sorted = self.sortieren(pop_merged)
            else:
                pop_sorted = self.sortieren(pop_merged, inverse = True)
                
            # Truncate Merged Population
            pop['Position'] = pop_sorted['Position'][0:self.n_pop]
            pop['Cost'] = pop_sorted['Cost'][0:self.n_pop]
        
            # Store Best Cost and Position
            if min_max == 'minimize':
                BestCost.append(pop['Cost'][0])
                BestPosition.append(pop['Position'][0])
            else:
                BestCost.append(pop['Cost'][-1])
                BestPosition.append(pop['Position'][-1])
                
            # Store Number of Function Evaluation
            nfe[it] = NFE
            
            print("Iteration : {},NFE: {}, Best Cost: {}".format(it, nfe[it], BestCost[it]))
            
        return nfe, BestCost, BestPosition
    
    # In this function the convergence process of GA is plotted.        
    def plot_convergence(self, nfe, best_cost):
        
        plt.figure(figsize = [15,8])
        plt.plot(nfe, best_cost)
        plt.xlabel('Number of Function Evaluation')
        plt.ylabel('Best Cost Function')
    
    # In this function a heatmap showing the values of chromosomes through the process of training GA is revealed.
    def heatmap(self, nfe, best_position):
        
        plt.figure(figsize = [15,8])
        sns.heatmap(np.array(best_position),cmap = "Blues", linewidths = 0.01)
        plt.xticks(np.arange(1, self.n_var + 1),np.arange(1, self.n_var + 1))
        plt.yticks(np.arange(1, len(best_position)+1,5),np.arange(1, len(best_position)+1,5))
        plt.xlabel('binary array')
        plt.ylabel('iteration')
        plt.grid()
        

