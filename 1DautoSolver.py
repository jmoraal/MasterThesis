# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:30:42 2022

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.integrate import solve_ivp

### Main todos
#TODO Find reasonable parameters
#TODO Use NaNs in position array to be more memory and time efficient (note that dist computation is O(n²)!)
#TODO make more class-based? May help a lot with readability etc


### Simulation settings
simTime = 1000         # Total simulation time
PBCs = False         # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
reg = 'cutoff'         # Regularisation technique; for now either 'eps' or 'cutoff' #TODO implement better regularisation, e.g. from Michiels20
eps = 0.01        # To avoid singular force making computation instable. 
boxLength = 6

def setExample(N, boxLen = 1): 
    global boxLength, initialPositions, b, creations, domain, initialNrParticles
    
    boxLength = boxLen
        
    if N == 0: ### Example 0: #TODO these trajectories are not smooth, seems wrong...
        initialPositions = np.array([0.3, 0.75]) # If they are _exactly_ 0.5 apart, PBC distance starts acting up; difference vectors are then equal ipv opposite
        b = np.array([1, 1])    
    elif N == 1: ### Example 1:
        initialPositions = np.array([0.21, 0.7, 0.8]) 
        b = np.array([-1, 1, 1])  
    elif N == 2: ### Example 2:
        initialPositions = np.array([0.02, 0.2, 0.8, 0.85, 1]) 
        b = np.array([-1, -1, 1, 1, -1]) # Particle charges
    else: ### Example 3: Arbitrary number of particles
        initialPositions = np.random.uniform(size = N, low = 0, high = boxLength)
        #charges:
        b = np.ones(N)
        neg = np.random.choice(range(N),N//2, replace=False)
        b[neg] = -1
        
    # Dependent paramaters (deducable from the ones defined above): 
    initialNrParticles = len(initialPositions)
    domain = (0,boxLength)

setExample(100, boxLen = boxLength)

### Create grid, e.g. as possible sources: 
# nrSources = 11
# sources = np.linspace(0,boxLength-1/(nrSources - 1), nrSources) # Remove last source, else have duplicate via periodic boundaries
# # sources = np.array([0.21, 0.3, 0.45, 0.75, 0.8])
# # nrSources = len(sources)
# nrBackgrSrc = 100 #Only to plot
# backgrSrc = np.linspace(0,boxLength-1/(nrBackgrSrc - 1), nrBackgrSrc)

# %%
#@jit(nopython=True)
def pairwiseDistance(x1, PBCs = True, x2 = None):
    """ Compute distances, optionally with PBCs
    
    Computes distances between all atoms, optionally for 
    closest copies taking boundaries into account.
    
    INPUT: np array of coordinates (single or multiple dimensions)
    OUTPUT: array of difference vectors and one of distances 
            between all coordinates, optionally w/ PBCs
    """    
    
    if x2 is None: 
        x2 = x1
    
    diff = x1 - x2[:,np.newaxis] # Difference vectors 
    if PBCs: # There may be an easier way in 1D...
        diff = diff - np.floor(0.5 + diff/boxLength)*boxLength # Calculate difference vector to closest copy of particle (with correct orientation). 
        
    dist = np.abs(diff) # Compute length of difference vectors (in 1D, else use linalg.norm w/ axis=2)
    
    return diff, dist




def f(t,x, regularisation = 'eps', PBCs = False):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    diff, dist = pairwiseDistance(x, PBCs = PBCs)
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to (arbitrary) non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2
    
    
    interactions = -(diff / distCorrected) * chargeArray #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'V1': 
        interactions[dist < eps] = diff/eps**2
    
    if regularisation == 'cutoff': 
        interactions = np.clip(interactions, -1/eps, 1/eps) # Sets all values outside [-c,c] to value of closest boundary
    
    updates = np.nansum(interactions,axis = 1) 
    
    return updates
    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x[np.isfinite(x)] %= boxLength # Also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension
    # Index by isfinite to not affect np.nan (still works otherwise, but gives warning)

    return x


# %% SIMULATION

simStartTime = timer.time() # To measure simulation computation time

### Simulation loop
sol = solve_ivp(f, [0, simTime], initialPositions, method = 'BDF') #BDF is best for problems that may be stiff

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %% Visualisation: 

def plotODESol(solution, charges, log = False): 
    plt.clf()
    t = sol.t
    x = np.transpose(sol.y)
        
    colorDict = {-1:'red', 0:'grey', 1:'blue'}
    nrParticles = len(x[0])
    plt.ylim((0,t[-1]))
    plt.xlim((np.min(x),np.max(x)))
    if log: 
        t[0] = t[1]/2 #So that first timestep is clearly visible in plot. Not quite truthful, but also not quite wrong. 
        plt.yscale('log')
        plt.ylim((1e-6,t[-1])) 
    
    
    for i in range(nrParticles) : 
        x_current = x[:,i]
        x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
        pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1
        x_new = np.insert(x_current, pos, np.nan)
        t_new = np.insert(t, pos, np.nan) # Insert NaNs in order not to draw jumps across domain caused by PBCs
        # Note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_new, t_new, c = colorDict.get(b[i]))
        
    plt.show()

plotODESol(sol, b, log = True)
