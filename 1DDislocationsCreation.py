# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:36:14 2022

@author: s161981
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:41:52 2021

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
import time as timer

### Main todos
#TODO Implement better integration scheme! Currently use forward Euler (originating from stochastic SDE approach) with improvised adaptive timestep
#TODO Find reasonable parameters
#TODO Use NaNs in position array to be more memory and time efficient (note that dist computation is O(n²)!)
#TODO adapt variable names to correspond to overleaf (or other way around)

#%% INITIALISATION

### Simulation settings
simTime = 0.5           # Total simulation time
dt = 0.001              # Timestep for discretisation (or maximum, if adaptive timestep)
adaptiveTime = True     # Whether to use adaptive timestep in integrator
PBCs = False            # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
reg = 'cutoff'          # Regularisation technique; for now either 'eps', 'V1' (after vMeurs15), 'cutoff' or 'none' (only works when collTres > 0)
eps = 0.01              # Regularisation parameter
cutoff = 200            # Regularisation parameter. Typically, force magnitude is 50 just before annihilation with eps=0.01
randomness = False      # Whether to add random noise to dislocation positions (normal distr.)
sigma = 0.01            # Standard dev. of noise (volatility)
withAnnihilation = True # Whether dislocations disappear from system after collision (under annihilation rules)
collTres = 1e-3         # Collision threshold; if particles are closer than this, they are considered collided. Should be Should be very close to 0 if regularisation is used
stress = 0              # External force (also called 'F'); only a constant in 1D case. Needed for creation in empty system
withCreation = True     # Whether dipoles are introduced automatically according to creation rule (as opposed to explicit time and place specification)
creaProc = 'zero'       # Creation procedure; either 'lin', 'zero' or 'dist' (for linear gamma, zero-gamma or distance creation respectively)
Fnuc = 10               # Threshold for magnitude of Peach-Koehler force 
tnuc = 0.001            # Threshold for duration of PK force magnitude before creation
drag = 1                # Multiplicative factor; only scales time I believe (and possibly external force)
showBackgr = False      # Whether to plot PK force field behind trajectories
domain = (0,1)          # Interval of space where initial dislocations and sources are placed (and possibly PBCs)



def setExample(N): 
    '''
    Initialises positions and charges of given nr of dislocations

    Parameters
    ----------
    N : int
        number of dislocations to initialise. 0,1,2,5 gives fixed examples, 
        other N give randomised settings with equally many +/- dislocations.

    Returns
    -------
    None (sets global variables).

    '''
    global initialPositions, initialCharges, initialNrParticles
    
    if N == 0: ### Empty example; requires some adaptions in code below (e.g. turn off 'no dislocations left' break)
        initialPositions = np.array([np.nan, np.nan]) 
        initialCharges = np.array([1, -1])    
    
    
    elif N == 1: ### Example 1:
        initialPositions = np.array([0.5]) 
        initialCharges = np.array([1])
        
        
    elif N == 2: ### Example 2: 
        initialPositions = np.array([0.3, 0.75]) # If they are _exactly_ 0.5 apart, PBC distance starts acting up; difference vectors are then equal ipv opposite
        initialCharges = np.array([1, -1])    
    
    
    elif N == 5: ### Example 3: 
        initialPositions = np.array([0.02, 0.2, 0.8, 0.85, 1]) 
        initialCharges = np.array([-1, -1, 1, 1, -1]) # Particle charges
    
        
    else: ### Given nr of particles, and option
        initialPositions = np.random.uniform(size = N) 
        initialCharges = np.ones(N)
        neg = np.random.choice(range(N),N//2, replace=False) #pick floor(N/2) indices at random
        initialCharges[neg] = -1 # Set half of charges to -1, rest remains 1
        # note: different from completely random charges, which would be the following: 
        # initialCharges = np.random.choice((-1,1),nrParticles)
    
    
    # Dependent paramaters (deducable from the ones defined above): 
    initialNrParticles = len(initialPositions)
    
        

def setSources(M, background = showBackgr): 
    '''
    Initialises positions and charges of given nr of sources

    Parameters
    ----------
    M : int
        number of sources to initialise (except -1). -1,1 gives fixed examples,
        -1 serving as fixed reference case with 5 non-evenly distributed sources.
        Other values give N sources evenly distributed on interval [0,1]
    background : boolean, optional
        indicates whether background sources should be initialised 
        (only if PK force field is plotted). The default is showBackgr.

    Returns
    -------
    None (sets global variables).

    '''
    global nrSources, sourceLocs, nrBackgrSrc, backgrSrc
    # Initialise sourceLocs for creation: 
    
    if M == -1: 
        sourceLocs = np.array([0.21, 0.3, 0.45, 0.75, 0.8])
    
    elif M == 1: 
        sourceLocs = np.array([0.55])
        
    
    else: # Evenly distribute given nr of sources
        if PBCs: # Remove last source, else have duplicate via periodic boundaries
            sourceLocs = np.linspace(0, 1-1/(M - 1), M) 
        else: sourceLocs = np.linspace(0, 1, M)
    
    
    nrSources = len(sourceLocs)
    
    if background: # Initialise additional sources for visualisation of PK-force: 
        nrBackgrSrc = 100 
        backgrSrc = np.linspace(0,1, nrBackgrSrc)


setExample(10)
if withCreation: 
    setSources(12)


# Additional comparison case (randomly generated but fixed): 
initialPositions = np.array([0.00727305, 0.04039581, 0.25157344, 0.2757077 , 0.28350536,
       0.36315111, 0.60467167, 0.68111491, 0.72468363, 0.7442808 ])
initialCharges = np.array([-1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.])
sourceLocs = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
nrSources = len(sourceLocs)
initialNrParticles = len(initialPositions)




# %% DEFINITIONS
def pairwiseDistance(x1, PBCs = False, x2 = None):
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
        diff = diff - np.floor(0.5 + diff) # Calculate difference vector to closest copy of particle (with correct orientation). 
        
    dist = np.abs(diff) # Compute length of difference vectors (in 1D, else use linalg.norm w/ axis=2)
    
    return diff, dist



def interaction(diff,dist,b, PBCBool = False, regularisation = 'eps'):
    '''
    Compute array of pairwise particle interactions for given array of particle coordinates and charges 

    Parameters
    ----------
    diff : 2D numpy array (nxn)
        pairwise differences between all particle positions.
    dist : 2D numpy array (nxn)
        pairwise distances between all particle positions.
    b : 1D numpy array (n)
        charges of dislocations.
    PBCBool : boolean, optional
        whether to use periodic boundaries. The default is False.
    regularisation : 'eps', 'V1', 'cutoff' or 'none', optional
        interaction regularisation method. The default is 'eps'.

    Returns
    -------
    interactions : 2D numpy array (nxn)
        representing force pairs of particles exert on each other.

    '''
    
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to (arbitrary) non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix (b_i b_j)_ij
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2 # Normalise to avoid computational problems at singularity. Square to normalise difference vector
        
        # interactions = -(1/diff) * chargeArray # Only in 1D; else need diff/dist
    
    interactions = -(diff / distCorrected) * chargeArray # Calculates matrix b_i b_j / (x_i - x_j)
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'V1': 
        interactions[dist < eps] = diff/eps**2
    
    if regularisation == 'cutoff': 
        interactions = np.clip(interactions, -cutoff, cutoff) # Sets all values outside [-c,c] to value of closest boundary
    
    return interactions



def projectParticles(x):
    """Projects particles into box of given size."""
    
    x[np.isfinite(x)] %= 1 # Also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension
    # Index by isfinite to not affect np.nan (still works otherwise, but gives warning)

    return x



def PeachKoehler(sourceLocs, x, b, stress, regularisation = 'eps'):
    """Computes Peach-Koehler force for each source, possibly for regularised interaction
    
    (Sources are typically fixed equidistant grid points) """
    
    x = np.nan_to_num(x) #sets NaNs to 0
    diff, dist = pairwiseDistance(x, x2 = sourceLocs) 
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2
    
    interactions = (diff / distCorrected) * b[np.newaxis,:] + stress # According to final expression in meeting notes of 211213
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    f = np.sum(interactions, axis = 1) # Per source, sum contributions over all dislocs
    
    return f
 


class Source: 
    def __init__(self, loc):
        self.pos = loc
        self.tAboveThreshold = 0
    
        
    def updateThresTime(self, PK, dt, N):
        if (np.abs(PK) > Fnuc): 
            self.tAboveThreshold += dt
            if self.tAboveThreshold >= tnuc: 
                self.tAboveThreshold = 0
                return True
        else: 
            self.tAboveThreshold = 0
        


class Creation: 
    def __init__(self, loc, PK, t, idx):
        self.loc = loc
        self.PKAtCrea = PK
        self.creaTime = t
        self.idx = idx
        self.inProgress = True
            
    
    def createDipole(self):
        
        if creaProc == 'lin': #TODO this function should probably not have an asymptote! 
            self.texc = 0.283465/(self.PKAtCrea-0.325376) + -0.013909/(self.PKAtCrea-0.325376)**2 + 0.000511/(self.PKAtCrea-0.325376)**3
            locs = np.array([self.loc - 0.5*collTres, self.loc + 0.5*collTres])
        
        elif creaProc == 'zero': 
            self.texc = 1/(2*np.abs(self.PKAtCrea))**2
            locs = np.array([self.loc - 0.5*collTres, self.loc + 0.5*collTres])
        
        elif creaProc == 'nuc':
            self.Lnuc = 1/(2* np.abs(self.PKAtCrea))
            locs = np.array([self.loc - 0.5*self.Lnuc, self.loc + 0.5*self.Lnuc])
        
        charges = np.array([1,-1])*np.sign(self.PKAtCrea)
        
        return locs, charges
    
    
    def forceAdjustment(self, t):
        if creaProc == 'lin': 
            forceFact = (2*(t - self.creaTime)/self.texc - 1)
        
        elif creaProc == 'zero': 
            forceFact = 0
        
        return forceFact
    
    
    def exceptionCheck(self, t): 
        
        if (creaProc == 'lin') or (creaProc == 'zero'): 
            if t > self.creaTime + self.texc: 
                self.inProgress = False 
        elif creaProc == 'nuc':
            self.inProgress = False


class Annihilation: 
    def __init__(self, annTime, dislocs):
        self.annTime = annTime
        self.dislocs = dislocs
        


# %% SIMULATION

### Precomputation
t = 0
stepSizes = []
times = [0]

trajectories = initialPositions[None,:] # Change shape into (1,len)
x = np.copy(initialPositions) # 'copy' to create new array; otherwise, x is just a reference (and changes if initialPositions changes)
b = np.copy(initialCharges) # 'copy' to create new array; otherwise, b is just a reference (and changes if initialCharges changes)

annihilations = [] # To keep track of all annihilation events (only for analysis afterwards)

if withCreation: # Initialise sources and store classes in list
    sources = [Source(x) for x in sourceLocs] # Initialise source classes and store in list
    creations = []  # List of _all_ creations occurring at some point in entire system (only for analysis afterwards)
    currentCreations = [] # List to keep track of creations for which exception holds
        
    if showBackgr: 
        PKlog = np.zeros((0,nrBackgrSrc))

simStartTime = timer.time() # To measure simulation computation time

### Simulation loop
while t < simTime: 
    # Creation: 
    if withCreation: # Idea: if force at source exceeds threshold for certain time, new dipole is created
        PK = PeachKoehler(sourceLocs, x, b, stress) 
        
        if showBackgr: 
            PKlog = np.append(PKlog, PeachKoehler(backgrSrc, x,b,stress)) # Save all PK forces to visualise
        
        locs = []
        charges = []
        
        for i,src in enumerate(sources): 
            thresReached = src.updateThresTime(PK[i], dt, len(x)) # Sources updated, boolean indicates whether threshold is reached
            if thresReached: # If threshold is reached, initiate creation procedure: 
                newCrea = Creation(src.pos, PK[i], t, len(x) + 2*len(creations)) # Initialise Creation
                creations.append(newCrea) # Append to  list of all creations
                currentCreations.append(newCrea) # Append to  list of current creations
                locs, charges = Creation.createDipole(newCrea) # Obtain corresponding dipole locations and charges
                x = np.append(x, locs) # Add new dipole to position vector
                b = np.append(b, charges) # Add new dipole to charge vector

                initialCharges = np.append(initialCharges, charges) # Store charges of new For correct plot colours
                trajectories = np.append(trajectories, np.zeros((len(trajectories), len(locs)))*np.nan, axis = 1) #extend _entire_ position array (over all timesteps) with NaNs. #TODO can we predict a maximum a priori? May be faster than repeatedly appending
        
        
        # Check whether creations still have exception: 
        for crea in currentCreations: 
            crea.exceptionCheck(t)
        
        # Remove Creations that have no force exception (anymore) from list: 
        currentCreations = [crea for crea in currentCreations if crea.inProgress]  
  
    
    # Compute main forces/interaction: 
    diff, dist = pairwiseDistance(x, PBCs = PBCs)
    
    
    if withAnnihilation: # Make dislocs annihilate when necessary (before computing interactions, s.t. annihilated dislocs indeed do not influence the system anymore)
        tempDist = np.nan_to_num(dist, nan = 1000) + 1000*np.tril(np.ones((len(x),len(x)))) 
        # Set nans and non-above-diagonal entries to arbitrary large number,  
        # so that effectively all entries on and below diagonal are disregarded in comparison below
        
        collPart1, collPart2 = np.where((tempDist < collTres)) # Identifies pairs closer than collTres together. Format: ([parts A], [parts B]).
        # First fix: combine all overlapping pairs into dislocations. But note that this can let dislocations annihilate from arbitrarily far away...
        # Note: looking at above-diagonal entries, all pairs are ordered w/ smallest index first. 
        # But problem: it actually depends on which index you take first which ones annihilate...
        
        for i in np.unique(collPart1): # Iterate over all particles involved in annihilation (1st in pair)
            dislocGroup = collPart2[collPart1 == i] #Find all other particles i is close enough to
            if len(dislocGroup) % 2 == 1:  #If even number of dislocs annihilate (so nr dislocs i annihilates with is odd), all end up with charge 0
                b[i] = 0
                x[i] = np.nan
            else: # In other cases, one disloc ends up with remaining charge and set to average position of annihilating dislocs.
                b[i] = np.sum(b[dislocGroup]) + b[i] #give i remaining charge (i is arbitrary choice; does not matter for solution)
                x[i] = np.average(np.append(x[dislocGroup], x[i]))
            
            b[dislocGroup] = 0 #Set charges of all others to 0
            x[dislocGroup] = np.nan #Remove from system
            
            annihilations.append(Annihilation(t, np.append(dislocGroup,i)))
    
    
    interactions = interaction(diff,dist,b, PBCBool = PBCs, regularisation = reg)
    
        
    # Adjust forces between newly created dislocations (keeping track of time since each creation separately)
    if withCreation: 
        for crea in currentCreations: # 'creations' now only contains pairs with force exception
            j = crea.idx
            interactions[j : j + 2, j : j + 2] *= crea.forceAdjustment(t)
        
    
    ## Main update step: 
    updates = np.nansum(interactions,axis = 1)  # Deterministic part; treating NaNs as zero
    
    if adaptiveTime: 
        dt = max(min(0.001/np.max(np.abs(updates)), dt),1e-10) # rudimentary adaptive timestep; always between (1e-10, dt)
        stepSizes.append(dt)
    
    x_new = x + drag * updates * dt # Alternative file available with built-in ODE-solver, but without creation
    
    if randomness: 
        random = sigma * np.random.normal(size = len(x)) # Random part of update
        x_new += random * np.sqrt(dt) 
    
    
    if PBCs: 
        x_new = projectParticles(x_new) # Places particles back into box
    
    trajectories = np.append(trajectories, x_new[None,:], axis = 0) # Store all positions for visualisation
    
    
    x = x_new
    t += dt
    times.append(t)
    
    
    # Stop simulation if all dislocations have annihilated:
    # (in this case PK=0, so no creations can occur anymore either)
    if np.isnan(x_new).all():
        print('No dislocations left')
        break # Breaks from outer 'while'-loop
    
    
    if((10*t/simTime) % 1 < 2*dt):
        print(f"{t:.5f} / {simTime}")


#Compute and print copmutation time for simulation 
duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %% VISUALISATION

#1D plot:    
def plot1D(bInitial, trajectories, t, PK = None, log = False): 
    global pos, x_temp
    plt.clf() # Clears current figure
    
    trajectories = np.ndarray.squeeze(trajectories)
    colorDict = {1:'red', 0:'grey', -1:'blue'}
    
    nrPoints, nrTrajectories = np.shape(trajectories)
    
    y = t
    plt.ylim((0,t[-1]))
    
    if log: # Plot with time on log-scale
        t[0] = t[1]/2 #So that first timestep is clearly visible in plot. Not quite truthful, but also not quite wrong. 
        plt.yscale('log')
        plt.ylim((t[0],t[-1])) 
    
    
    for i in range(nrTrajectories):
        x_current = trajectories[:,i]
        y_current = y
        if PBCs: #insert NaNs at 'discontinuities':
            x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
            pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1 # find jumps across domain caused by PBCs
            x_current = np.insert(x_current, pos, np.nan) # Insert NaNs in order not to draw jumps
            y_current = np.insert(y, pos, np.nan) 
            # Note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_current, y_current, c = colorDict.get(bInitial[i]))
        # Set colour of created dislocations according to charge they eventually get (not 0, which they begin with)
        
    if not PK is None:
        # Broadcast timesteps and locations of sources into same size arrays:
        timeCoord = y[:,np.newaxis] * np.ones(len(PK[0]))
        locCoord = backgrSrc * np.ones(len(PK))[:,np.newaxis]
        
        if showBackgr: 
            PKnew = np.log(np.abs(PK) + 0.01) / np.log(np.max(np.abs(PK))) # Scale to get better colourplot
            
            plt.scatter(locCoord, timeCoord, s=50, c=PKnew, cmap='Greys')
        #May be able to use this? https://stackoverflow.com/questions/10817669/subplot-background-gradient-color/10821713



def printSummary(): 
    global nrRecollided, annDislocs, creaDislocs, overlap
    print("Total nr of Creations: ", len(creations))
    if t < simTime: 
        print("Total annihilation time: ", t)
    else: 
        print("Total annihilation time: not reached")
    
    
    # See how many created dipoles recollided: 
    annDislocs = np.zeros((len(annihilations),2)) #NOTE: may not work if three or more dislocs annihilate! 
    creaDislocs = np.zeros((len(creations),2))
    nrRecollided = 0
    for i,ann in enumerate(annihilations): 
        annDislocs[i,:] = np.sort(ann.dislocs) #sort on lower index first to ease comparison below
    
    for i,crea in enumerate(creations): 
        creaDislocs[i,:] = np.array([crea.idx, crea.idx + 1])
    
    overlap = [pair for pair in creaDislocs if pair in annDislocs] # Create list of pairs that were created, but also annihilated
    # Probably also possible without for-loop (but maybe not worth the time to come up with that)
    nrRecollided = len(overlap)
    
    print(len(creations), " creations, ", 
          len(creations) - nrRecollided, " survived. (i.e. ", 
          100*(1-nrRecollided/len(creations)), "%)")
    
    
    

if showBackgr: 
    plot1D(initialCharges, trajectories, times, PK = PKlog)
else: 
    plot1D(initialCharges, trajectories, times, log = False)

printSummary()
