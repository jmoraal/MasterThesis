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
simTime = 1         # Total simulation time
dt = 0.0005         # Timestep for discretisation
PBCs = False         # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
reg = 'eps'         # Regularisation technique; for now either 'eps' or 'cutoff' #TODO implement better regularisation, e.g. from Michiels20
eps = 5e-7        # To avoid singular force making computation instable. 
cutoff = 50         # To avoid singular force making computation instable. 
randomness = False  # Whether to add random noise to dislocation positions
sigma = 0.01        # Influence of noise
sticky = True       # Whether collisions are sticky, i.e. particles stay together once they collide (can probably be made standard)
collTres = 0.005    # Collision threshold; if particles are closer than this, they are considered collided
manualCrea = False  # Whether to include manually appointed dislocation creations
creaExc = 0.2       # Time for which exception rule governs interaction between newly created dislocations. #TODO now still arbitrary threshold.
stress = 100          # Constant in 1D case. Needed for creation
autoCreation = True # Whether dipoles are introduced according to rule (as opposed to explicit time and place specification)
forceTres = 1000    # Threshold for magnitude Peach-Koehler force
timeTres = 0.02     # Threshold for duration of PK force magnitude before creation
Lnuc = 2*collTres # Distance at which new dipole is introduced. Must be larger than collision threshold, else dipole annihilates instantly
showBackgr = False 
autoSolve = True


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

setExample(6)

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




def f(t,x):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    diff, dist = pairwiseDistance(x, PBCs = PBCs)
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to (aritrary) non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    
    distCorrected = dist**2 + eps
    
    interactions = -1/len(b) * (diff / distCorrected) * chargeArray #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    updates = np.nansum(interactions,axis = 1) 
    
    return updates
    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x[np.isfinite(x)] %= boxLength # Also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension
    # Index by isfinite to not affect np.nan (still works otherwise, but gives warning)

    return x



def PeachKoehler(sources, x, b, stress, regularisation = 'eps'):
    """Computes Peach-Koehler force for each source
    
    (Sources are typically fixed equidistant grid points) """
    
    x = np.nan_to_num(x)
    dist, diff = pairwiseDistance(x, x2 = sources) 
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2
    
    
    interactions = -1/len(b) * (diff / distCorrected) * b[np.newaxis,:] #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'eps': 
        dist = (dist + eps) # Normalise to avoid computational problems at singularity #TODO doubt it is right to use same eps here!
    
    
    interactions = 2/dist + stress # According to final expression in meeting notes of 211213
    f = np.sum(interactions, axis = 1)
    
    return f
        

# %% SIMULATION

simStartTime = timer.time() # To measure simulation computation time

### Simulation loop
sol = solve_ivp(f, [0, simTime], initialPositions, method = 'BDF') #BDF is best for problems that may be stiff

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %%

#1D plot:    
def plot1D(bInitial, trajectories, endTime, PK = None): 
    global pos, x_temp
    plt.clf() # Clears current figure
    
    trajectories = np.ndarray.squeeze(trajectories)
    colorDict = {-1:'red', 0:'grey', 1:'blue'}
    
    nrPoints, nrTrajectories = np.shape(trajectories)
    
    plt.ylim((0,endTime))
    y = np.linspace(0, endTime, nrPoints)
    
    for i in range(nrTrajectories):
        #insert NaNs at 'discontinuities':
        x_current = trajectories[:,i]
        x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
        pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1
        x_new = np.insert(x_current, pos, np.nan)
        y_new = np.insert(y, pos, np.nan) # Insert NaNs in order not to draw jumps across domain caused by PBCs
        # Note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_new, y_new, c = colorDict.get(bInitial[i]))
        # Set colour of created dislocations according to charge they eventually get (not 0, which they begin with)
        
    # if not PK is None:
    #     # Broadcast timesteps and locations of sources into same size arrays:
    #     timeCoord = y[:,np.newaxis] * np.ones(len(PK[0]))
    #     locCoord = backgrSrc * np.ones(len(PK))[:,np.newaxis]
        
    #     PKnew = np.log(np.abs(PK) + 0.01) / np.log(np.max(np.abs(PK))) # Scale to get better colourplot
        
    #     plt.scatter(locCoord, timeCoord, s=50, c=PKnew, cmap='Greys')
    #     #May be able to use this? https://stackoverflow.com/questions/10817669/subplot-background-gradient-color/10821713

def plotODESol(solution, charges): 
    plt.clf()
    t = sol.t
    x = np.transpose(sol.y)
        
    colorDict = {-1:'red', 0:'grey', 1:'blue'}
    nrParticles = len(x[0])
    plt.ylim((0,t[-1]))
    plt.xlim((0,boxLength))
    
    for i in range(nrParticles) :
        x_current = x[:,i]
        x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
        pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1
        x_new = np.insert(x_current, pos, np.nan)
        t_new = np.insert(t, pos, np.nan) # Insert NaNs in order not to draw jumps across domain caused by PBCs
        # Note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_new, t_new, c = colorDict.get(b[i]))
        
    plt.show()
