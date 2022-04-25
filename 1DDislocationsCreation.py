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
#TODO make sticky collisions faster! Now very inefficient
#TODO adapt variable names to correspond to overleaf (or other way around)

#%% INITIALISATION

### Simulation settings
simTime = 0.5       # Total simulation time
dt = 0.0005         # Timestep for discretisation (or initial, if adaptive timestep)
adaptiveTime = True # Whether to use adaptive timestep in integrator
PBCs = False        # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
reg = 'none'        # Regularisation technique; for now either 'eps', 'V1' (after vMeurs15), 'cutoff' or 'none' (only works when collTres > 0)
eps = 0.01          # Regularisation parameter
cutoff = 50         # Regularisation parameter. Typically, force magnitude is 50 just before annihilation with eps=0.01
randomness = False  # Whether to add random noise to dislocation positions (normal distr.)
sigma = 0.01        # Standard dev. of noise (volatility)
sticky = True       # Whether collisions are sticky, i.e. particles stay together once they collide (can probably be made standard)
collTres = 0.005    # Collision threshold; if particles are closer than this, they are considered collided
manualCrea = False  # Whether to include manually appointed creation events
creaExc = 0.1       # Time for which exception rule governs interaction between newly created dislocations. #TODO now still arbitrary threshold; should be adaptive!
stress = 0          # External force (also called 'F'); only a constant in 1D case. Needed for creation in empty system
autoCreation = True # Whether dipoles are introduced automatically according to creation rule (as opposed to explicit time and place specification)
creaProc = 'zero'    # Creation procedure; either 'lin', 'zero' or 'dist' (for linear gamma, zero-gamma or distance creation respectively)
Fnuc = 0.5    # Threshold for magnitude of Peach-Koehler force 
tnuc = 0.05     # Threshold for duration of PK force magnitude before creation
Lnuc = 2*collTres   # Distance at which new dipole is introduced. Must be larger than collision threshold, else dipole annihilates instantly
drag = 1            # Multiplicative factor; only scales time I believe (and possibly external force)
showBackgr = False  # Whether to plot PK force field behind trajectories
domain = (0,1)      # Interval of space where initial dislocations and sources are placed (and possibly PBCs)


def setExample(N): 
    global initialPositions, b, creations, domain, initialNrParticles
        
    if N == -1: ### Empty example; requires some adaptions in code below (e.g. turn off 'no dislocations left' break)
        initialPositions = np.array([np.nan, np.nan]) 
        b = np.array([1, -1])    
    
    elif N == 0: ### Example 0: 
        initialPositions = np.array([0.3, 0.75]) # If they are _exactly_ 0.5 apart, PBC distance starts acting up; difference vectors are then equal ipv opposite
        b = np.array([1, -1])    
    
    elif N == 1: ### Example 1:
        initialPositions = np.array([0.21, 0.7, 0.8]) 
        b = np.array([-1, 1, 1])
        
    
    elif N == 2: ### Example 2: manually appointed creation events
        initialPositions = np.array([0.02, 0.2, 0.8, 0.85, 1]) 
        b = np.array([-1, -1, 1, 1, -1]) # Particle charges
        # Creation time, place and 'orientation' (+/- charge order):
        creations = np.array([[0.14, 0.5, 1, -1],
                              [0.3, 0.2, 1, -1],
                              [0.6, 0.5, 1, -1],
                              [0.61, 0.1, 1, -1]])
        # Format: time, loc, orient is (0.15, [0.5], [-1,1]). Not great, but easiest to keep overview for multiple creations
        #TODO seems like having two creations occur at exact same time is not yet possible
    
    elif N == 3: ### Example 3: # 10 randomly distributed dislocations (5+,5-), with arbitrary (but 'manual') creation events
        nrParticles = 10
        initialPositions = np.random.uniform(size = nrParticles, low = 0, high = 1)
        # 0/1 charges:
        b = np.random.choice((-1,1),nrParticles)
        
        nrCreations = 5
        creations = np.zeros((nrCreations,4))
        creations[:,0] = np.linspace(0,0.5*simTime, nrCreations) # Creation times; Last creation at 0.5 simTime to see equilibrium develop
        creations[:,1] = np.random.uniform(size = nrCreations, low = 0, high = 1) # Locations. Needs to be adapted for multi-D
        creations[:,2:] = np.random.choice((-1,1),nrCreations)[:,np.newaxis] * np.array([1,-1]) # Creation orientation, i.e. order of +/- charges
    
    else: 
        initialPositions = np.random.uniform(size = N) 
        b = np.ones(N)
        neg = np.random.choice(range(N),N//2, replace=False)
        b[neg] = -1 # Set half of charges to -1, rest remains 1
    
    
    # Dependent paramaters (deducable from the ones defined above): 
    initialNrParticles = len(initialPositions)

setExample(0)

### Create grid, e.g. as possible sources: 
nrSources = 11
sources = np.linspace(0,1-1/(nrSources - 1), nrSources) # Remove last source, else have duplicate via periodic boundaries
# sources = np.array([0.49])
# sources = np.array([0.21, 0.3, 0.45, 0.75, 0.8])
# nrSources = len(sources)
nrBackgrSrc = 100 #Only to plot
backgrSrc = np.linspace(0,1-1/(nrBackgrSrc - 1), nrBackgrSrc)


# %% FUNCTIONS
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
        diff = diff - np.floor(0.5 + diff) # Calculate difference vector to closest copy of particle (with correct orientation). 
        
    dist = np.abs(diff) # Compute length of difference vectors (in 1D, else use linalg.norm w/ axis=2)
    
    return diff, dist


def interaction(diff,dist,b, PBCBool = True, regularisation = 'eps'):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges 
    
    Also returns array of pairwise charge products; not strictly necessary, but speeds up computation
    """
    
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to (aritrary) non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2 # Normalise to avoid computational problems at singularity. Square to normalise difference vector
        
        # interactions = -(1/diff) * chargeArray # Only in 1D; else need diff/dist
    interactions = -(diff / distCorrected) * chargeArray 
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'V1': 
        interactions[dist < eps] = diff/eps**2
    
    if regularisation == 'cutoff': 
        interactions = np.clip(interactions, -cutoff, cutoff) # Sets all values outside [-c,c] to value of closest boundary
    
    return interactions, chargeArray

    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x[np.isfinite(x)] %= 1 # Also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension
    # Index by isfinite to not affect np.nan (still works otherwise, but gives warning)

    return x



def PeachKoehler(sources, x, b, stress, regularisation = 'eps'):
    """Computes Peach-Koehler force for each source, possibly for regularised interaction
    
    (Sources are typically fixed equidistant grid points) """
    
    x = np.nan_to_num(x) #sets NaNs to 0
    dist, diff = pairwiseDistance(x, x2 = sources) 
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps**2) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2
    
    interactions = (diff / distCorrected) * b[np.newaxis,:] + stress # According to final expression in meeting notes of 211213
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    f = np.sum(interactions, axis = 1) # Per source, sum contributions over all dislocs
    
    return f
        

# %% SIMULATION

### Precomputation
#nrSteps = int(simTime/dt) # Rounds fraction down to integer (floor)
t = 0
# if manualCrea: 
#     nrCreations = 2 * len(creations) # At every creation time, two new dislocations are introduced
#     nrParticles = initialNrParticles + nrCreations 
#     creationSteps = np.append(np.floor(creations[:,0]/dt),0) #append 0 to 'know' last creation has happend
#     creationCounter = 0
#     stepsSinceCreation = np.ones(len(creations)) * -1 # Set to -1 to indicate creation has not happened yet; set to 0 at moment of creation
#     exceptionSteps = int(creaExc / dt)
#     b = np.append(b, np.zeros(nrCreations))
# else: nrParticles = initialNrParticles
stepSizes = []
times = [0]

trajectories = initialPositions[None,:] # Change shape into (1,len)
x = np.copy(initialPositions)
#x[0,initialNrParticles:] = np.nan # Effectively keeps particles out of system before creation

bInitial = np.copy(b) #copy to create new array; otherwise, bInitial is just a reference (and changes if b changes)

if autoCreation: 
    tresHist = np.zeros(len(sources)) #threshold history; to measure how long the threshold was exceeded at certain source
    creaTrackers = []
    creaIdx = np.array([], dtype = 'int64')
    #exceptionSteps = int(creaExc / dt)
    if creaProc == 'lin': 
        excTimes = []
    
    if showBackgr: 
        PKlog = np.zeros((0,nrBackgrSrc))

simStartTime = timer.time() # To measure simulation computation time

### Simulation loop
while t < simTime: 
    # Creation: 
    if autoCreation: # Idea: if force at source exceeds threshold for certain time, new dipole is created
        PK = PeachKoehler(sources, x, b, stress) 
        if showBackgr: 
            PKlog = np.append(PKlog, PeachKoehler(backgrSrc, x,b,stress)) # Save all PK forces to visualise

        tresHist[np.abs(PK) >= Fnuc] += dt # Increment all history counters reaching Peach-Koehler force threshold by dt
        tresHist[np.abs(PK) < Fnuc] = 0 # Set all others to 0
        creations = np.where(tresHist >= tnuc)[0] # Identify which sources reached time threshold (automatically reached force threshold too). [0] to 'unpack' array from tuple
        nrNewCreations = len(creations)
        PKNewCreations = np.abs(PK[creations])
        nrNewDislocs = nrNewCreations * 2
        if nrNewDislocs > 0:
            tresHist[creations] = 0 # Reset counters of new creations
            
            # Set counters for exception time, to keep track of force exception:
            if creaProc == 'lin': #TODO need zeros of Rcrit from ODEPhaseplot here! 
                newExcTimes = 1/(4*PKNewCreations**2) #Now preliminary fix by taking same texc as for zero-gamma
            elif creaProc == 'zero': 
                newExcTimes = 1/(4*PKNewCreations**2)
            
            if (creaProc == 'lin' or creaProc == 'zero'): 
                creaTrackers = np.append(creaTrackers, newExcTimes) # Set counters to count down from exception time
                creaIdx = np.append(creaIdx, len(x) + np.arange(nrNewCreations)*2) # Keep track for which dislocations these are; keep 1st index of every pair
            if (creaProc == 'lin'): # For linear gamma creation, additionally:
                excTimes = np.append(excTimes, newExcTimes) # store exception times for linear gamma (actual function depends on this)
            
            #The following is a bit tedious, should be possible in a simpler way
            locs = np.zeros(nrNewDislocs)
            if creaProc == 'nuc': 
                locs[::2] = sources[creations] - 0.5*Lnuc # Read as: every other element 
                locs[1::2] = sources[creations] + 0.5*Lnuc # Read as: every other element, starting at 1
            else: # for gamma-creation, should have creation in exact same location, but this is computationally impossible. 
                  #Instead, since we annihilate at collTres, we also take this for creation
                locs[::2] = sources[creations] - 0.5*collTres 
                locs[1::2] = sources[creations] + 0.5*collTres 
            
            charges = np.zeros(nrNewDislocs)
            
            for i in range(nrNewCreations):
                sign = np.sign(PK[creations[i]]) # Index from 'creations' because PK is computed for all sources, not only new creations
                charges[2*i] = sign #TODO check sign! Might be other way around
                charges[2*i+1] = -sign 
            
            
            x = np.append(x, locs) # replace added NaNs by creation locations for current timestep
            b = np.append(b, charges)
            bInitial = np.append(bInitial, charges) #For correct plot colours
            trajectories = np.append(trajectories, np.zeros((len(trajectories), nrNewDislocs))*np.nan, axis = 1) #extend _entire_ position array (over all timesteps) with NaNs. #TODO can we predict a maximum a priori? May be faster than repeatedly appending
            
    #TODO case statement?    
        
    # if manualCrea and (k == creationSteps[creationCounter]): 
    #     creationLocation = creations[creationCounter][1]
    #     creationOrder = creations[creationCounter][-2:]
        
    #     idx = initialNrParticles + 2 * creationCounter
    #     x[k, idx] = creationLocation - 0.5*Lnuc # Set location, distance of collision treshold apart
    #     x[k, idx + 1] = creationLocation + 0.5*Lnuc 
    #     b[idx : idx + 2] = creationOrder # Set charges as specified before
    #     bInitial[idx : idx + 2] = creationOrder
        
    #     stepsSinceCreation[creationCounter] = 0
    #     creationCounter += 1
    #     print(f"Creation {creationCounter} took place")
    
    # main forces/interaction: 
    diff, dist = pairwiseDistance(x, PBCs = PBCs)
    interactions, chargeArray = interaction(diff,dist,b, PBCBool = PBCs, regularisation = reg)
    
        
    # Adjust forces between newly created dislocations (keeping track of time since each creation separately)
    if autoCreation and len(creaTrackers) > 0: 
        if (creaProc == 'lin' or creaProc == 'zero'): 
            for i in range(len(creaTrackers)): # For each recent creation...
                idx = initialNrParticles + 2 * creaIdx[i] #... select corresponding particle index
                
                # Multiply interactions within new dipole by corresponding gamma(t) from report:
                if creaProc == 'lin': 
                    interactions[idx : idx + 2,idx : idx + 2] *= 1.0 - 2*creaTrackers[i]/excTimes[i] # linear transition from opposite force (factor -1) to actual force (1)
                if creaProc == 'zero': 
                    interactions[idx : idx + 2,idx : idx + 2] = 0
            
            creaTrackers -= dt # Count down at all trackers
            creaIdx = creaIdx[creaTrackers > 0] # And remove corresponding indices, so that Idx[i] still corresponds to Tracker[i]
            if creaProc == 'lin': 
                excTimes = excTimes[creaTrackers > 0] # Also remove corresponding exception times
            creaTrackers = creaTrackers[creaTrackers > 0] # Remove those from list that reach 0
        
    
    # if manualCrea:
    #     #TODO do we also need this for automatic creations, or is relation w/ shear stress etc. enough?
    #     for i in range(len(creations)): #TODO unnecessarily time-consuming. Should be doable without loop (or at least not every iteration)
    #         if (0 <= stepsSinceCreation[i] < exceptionSteps+1): # Idea: disable forces between new creation initially
    #             idx = initialNrParticles + 2 * i
    #             # Idea: make interaction forces slowly transition from -1 (opposite) to 1 (actual force)
    #             interactions[idx : idx + 2,idx : idx + 2] *= -1.0 + 2*stepsSinceCreation[i]/exceptionSteps #1 - 2/(stepsSinceCreation[i] + 1) #TODO now still assumes creations are given in order of time. May want to make more robust
    #             stepsSinceCreation[i] += 1
    
    
    ## Main update step: 
    updates = np.nansum(interactions,axis = 1)  # Deterministic part; treating NaNs as zero
    if adaptiveTime: 
        dt = max(min(0.001/np.max(updates), 0.01),1e-10) # rudimentary adaptive timestep
        stepSizes.append(dt)
    x_new = x + drag * updates * dt # Alternative file available with built-in ODE-solver, but without creation
    
    # Stop simulation if all dislocations have annihilated:
    # (in this case PK=0, so no creations can occur anymore either)
    if np.isnan(x_new).all():
        print('No dislocations left')
        break
    
    
    if randomness: 
        random = sigma * np.random.normal(size = len(x)) # 'noise' part
        x_new += random * np.sqrt(dt) 
    
    
    if PBCs: 
        x_new = projectParticles(x_new) # Places particles back into box
    
    
    if sticky: 
        newDist = np.nan_to_num(pairwiseDistance(x_new, PBCs = PBCs)[1], nan = 1000) #TODO work around this, really don't want to calculate all distances twice!
        # Sets nan to arbitrary large number
        
        # Idea: dist matrix is symmetric and we don't want the diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        newDist += 1000*np.tril(np.ones((len(x_new),len(x_new)))) # Arbitrary large number, so that effectively all entries on and below diagonal are disregarded #TODO should this be x or x_new?
        collidedPairs = np.where((newDist < collTres) & (chargeArray == -1)) # Format: ([parts A], [parts B]). Makes sure particles only annihilate if they have opposite charges
        #TODO may want something nicer than this construction... 
        
        
        b[collidedPairs[0]] = 0 # If non-integer charges: take (b[collidedPairs[0]] + b[collidedPairs[1]])/2
        b[collidedPairs[1]] = 0
        x_new[collidedPairs[0]] = np.nan # Take annihilated particles out of system
        x_new[collidedPairs[1]] = np.nan 
        
    
    trajectories = np.append(trajectories, x_new[None,:], axis = 0)
    
    x = x_new
    t += dt
    times.append(t)
    
    if((10*t/simTime) % 1 < 1e-6):
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
        #insert NaNs at 'discontinuities':
        x_current = trajectories[:,i]
        x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
        pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1
        x_new = np.insert(x_current, pos, np.nan)
        y_new = np.insert(y, pos, np.nan) # Insert NaNs in order not to draw jumps across domain caused by PBCs
        # Note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_new, y_new, c = colorDict.get(bInitial[i]))
        # Set colour of created dislocations according to charge they eventually get (not 0, which they begin with)
        
    if not PK is None:
        # Broadcast timesteps and locations of sources into same size arrays:
        timeCoord = y[:,np.newaxis] * np.ones(len(PK[0]))
        locCoord = backgrSrc * np.ones(len(PK))[:,np.newaxis]
        
        if showBackgr: 
            PKnew = np.log(np.abs(PK) + 0.01) / np.log(np.max(np.abs(PK))) # Scale to get better colourplot
            
            plt.scatter(locCoord, timeCoord, s=50, c=PKnew, cmap='Greys')
        #May be able to use this? https://stackoverflow.com/questions/10817669/subplot-background-gradient-color/10821713

if showBackgr: 
    plot1D(bInitial, trajectories, times, PK = PKlog)
else: 
    plot1D(bInitial, trajectories, times, log = False)
