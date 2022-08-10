# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:30:42 2022

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.integrate import solve_ivp


# %% Simulation settings
simTime = 0.1         # Total simulation time
reg = 'none'         # Regularisation technique; for now either 'eps' or 'cutoff' 
eps = 0.01        # To avoid singular force making computation instable. 
boxLength = 1       # Domain width for initial coniguration
annihilation = True # Simulate with/without annihilation upon collision
collTres = 1e-4 # Collision threshold



def setExample(N, boxLen = 1): 
    '''
    Sets initial configuration (location and charge) as global parameters  
    for 3 minimal examples (N = 0,1,2) or any given number of dislocations (N > 2)

    '''
    global boxLength, initialPositions, b, creations, domain, initialNrParticles
    
    boxLength = boxLen
        
    if N == 0: ### Example 0: 
        initialPositions = np.array([-0.1, 0.1]) 
        b = np.array([1, 1])    
    elif N == 1: ### Example 1:
        initialPositions = np.array([0.21, 0.7, 0.8]) 
        b = np.array([-1, 1, 1])  
    elif N == 2: ### Example 2:
        initialPositions = np.array([0.02, 0.2, 0.8, 0.85, 1]) 
        b = np.array([-1, -1, 1, 1, -1]) # Particle charges
    elif N == 4: 
        initialPositions = np.array([-2,-1,1,2]) 
        b = np.array([-1, 1, -1, 1]) 
    elif N == 6: 
        initialPositions = np.array([-2,-1,-0.7, 0.7,1,2]) 
        b = np.array([-1, 1,-1, 1, -1, 1]) 
    else: ### Example 3: Arbitrary number of particles
        initialPositions = np.random.uniform(size = N, low = 0, high = boxLength)
        #charges:
        b = np.ones(N)
        neg = np.random.choice(range(N),N//2, replace=False)
        b[neg] = -1
        
    # Dependent paramaters (deducable from the ones defined above): 
    initialNrParticles = len(initialPositions)
    domain = (0,boxLength)


setExample(50, boxLen = boxLength)

# To plot: 
bInitial = np.copy(b)
annTimes = np.zeros(len(b))


# %% FUNCTIONS

def pairwiseDistance(x1, PBCs = True, x2 = None):
    """ Compute distances between all given coordinates, 
    optionally with PBCs and optionally to a second given set of coordinates
        
    INPUT: array of coordinates (single or multiple dimensions)
           (opt) boolean PBCs, whether to use periodic boundary conditions
           (opt) second coordinates array, computes pairwise distance between 
    OUTPUT: array of difference vectors and one of distances 
            between all coordinates
    """    
    
    if x2 is None: 
        x2 = x1
    
    diff = x1 - x2[:,np.newaxis] # Difference vectors 
    if PBCs: # Calculate difference vector to closest copy of particle 
        diff = diff - np.floor(0.5 + diff/boxLength)*boxLength 
        
    dist = np.abs(diff) # Length of difference vectors (in 1D)
    
    return diff, dist



def f(t,x, regularisation = 'eps', PBCs = False):
    """ Compute discrete gradient flow for given array of coordinates 
    according to Equation (9.1) from vMeurs15
    
    uses global parameters b (charges), eps, and 
    updates global parameters b and annTimes 
    
    t is dummy argument used by ODE solver"""
    global b2
    diff, dist = pairwiseDistance(x, PBCs = PBCs)
    # Set distance from particle to itself to (arbitrary) non-zero value to 
    # avoid div by 0; arbitrary, since this term cancels out anyway
    np.fill_diagonal(dist, 1) 
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) 
    else: 
        distCorrected = dist**2
    
    
    interactions = -(diff / distCorrected) * chargeArray #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'V1': 
        interactions[dist < eps] = diff/eps**2
    
    if regularisation == 'cutoff': # Set all values outside [-c,c] to closest boundary
        interactions = np.clip(interactions, -1/eps, 1/eps) 
    
    
    updates = np.nansum(interactions,axis = 1) 
    
    if annihilation: 
        collPart1, collPart2 = np.where((dist < collTres) * (chargeArray == -1)) 
        # Identifies pairs with opposite charge closer than collTres together 
        #('*' works as 'and' for 0/1 booleans). Format: ([parts A], [parts B]).
        
        bOld = np.copy(b) #to compare below, seeing whether annihilation happened
        
        # Go through list sequentially, annihilating not-yet-annihilated dislocs. 
        for i in range(len(collPart1)): 
            i1 = collPart1[i]
            i2 = collPart2[i]
            if b[i1] != 0 and b[i2] != 0 and i1 != i2: 
                b[i1] = 0
                b[i2] = 0
        
        annTimes[bOld != b] = t #save for plotting
        
        
    return updates



# %% SIMULATION

simStartTime = timer.time() # To measure simulation computation time

### Simulation: 
sol = solve_ivp(f, [0, simTime], initialPositions, method = 'BDF', max_step = 0.01) 
# (BDF is best for problems that may be stiff)

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %% VISUALISATION: 

def plotODESol(solution, charges, log = False, annihilationTimes = None): 
    ''' Given a solution from solve_ivp, plots trajectories according to given charges. 
    Optional: 
     - plot on log-scale if log = True
     - stop plotting if all dislocations annihilated    
    '''
    
    plt.clf()
    t = sol.t
    x = np.transpose(sol.y)
        
    colorDict = {1:'red', 0:'grey', -1:'blue'}
    nrParticles = len(x[0])
    plt.ylim((0,t[-1]))
    plt.xlim((np.min(x[np.isfinite(x)]),np.max(x[np.isfinite(x)])))
    if log: 
        t[0] = t[1]/2 #So that first timestep is clearly visible (avoid log(0))
        plt.yscale('log')
        plt.ylim((1e-6,t[-1])) 
    
    
    for i in range(nrParticles) : 
        if annihilationTimes is None: 
            t_current = t
            x_current = x[:,i]
        else: # Only plot trajectories up to annihilation
            t_current = t[t < annTimes[i]]
            x_current = x[:len(t_current),i]
        
        x_new = x_current
        t_new = t_current
        
        plt.plot(x_new, t_new, c = colorDict.get(charges[i]))
    
    plt.xlabel('$x$')
    plt.ylabel('$t$')
        
    plt.show()

if annihilation: 
    plotODESol(sol, bInitial, log = False, annihilationTimes=annTimes)
else: 
    plotODESol(sol, bInitial, log = False)
