# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:41:52 2021

@author: s161981
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as timer


### Simulation settings
simTime = 0.2         # Total simulation time
dt = 0.001          # Timestep for discretisation
PBCs = True         # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
reg = 'eps'         # Regularisation technique; for now either 'eps' or 'cutoff' #TODO implement better regularisation, e.g. from Michiels20
eps = 0.0001        # To avoid singular force making computation instable. 
cutoff = 50         # To avoid singular force making computation instable. 
randomness = False  # Whether to add random noise to dislocation positions
sigma = 0.01        # Influence of noise
sticky = True       # Whether collisions are sticky, i.e. particles stay together once they collide
collTres = 0.005    # Collision threshold; if particles are closer than this, they are considered collided
manualCrea = False   # Whether to include manually appointed dislocation creations
creaExc = 0.2       # Time for which exception rule governs interaction between newly created dislocations. #TODO now still arbitrary threshold.
stress = 1          # Constant in 1D case. Needed for creation
autoCreation = True # Whether dipoles are introduced according to rule (as opposed to explicit time and place specification)
forceTres = 50    # Threshold for magnitude Peach-Koehler force
timeTres = 0.02     # Threshold for duration of PK force magnitude before creation
Lnuc = 0.6*collTres # Distance at which new dipole is introduced

def setExample(N): 
    global boxLength, initialPositions, b, creations, domain, initialNrParticles
        
    if N == 1: ### Example 1:
        boxLength = 1
        initialPositions = np.array([[0.1], [0.85]]) 
        b= np.array([-1, 1])
        
    
    if N == 2: ### Example 2:
        boxLength = 1
        initialPositions = np.array([0.02, 0.2, 0.8, 0.85, 1]) 
        b = np.array([-1, -1, 1, 1, -1]) # Particle charges
        # Creation time, place and 'orientation' (positive/negative charge order):
        creations = np.array([[0.14, 0.5, 1, -1],
                              [0.3, 0.2, 1, -1],
                              [0.6, 0.5, 1, -1],
                              [0.61, 0.1, 1, -1]])
        # Format: time, loc, orient. is (0.15, [0.5], [-1,1]). Not great, but easiest to keep overview for multiple creations
        #TODO seems like two creations at exact same time are not yet possible
        # Also, orientation is quite important for whether a creation immediately annihilates or not
    
    
    if N == 3: ### Example 3:
        boxLength = 1
        nrParticles = 10
        initialPositions = np.random.uniform(size = nrParticles, low = 0, high = boxLength)
        # 0/1 charges:
        b = np.random.choice((-1,1),nrParticles)
        
        nrCreations = 5
        creations = np.zeros((nrCreations,4))
        creations[:,0] = np.linspace(0,0.5*simTime, nrCreations) # Creation times; Last creation at 0.5 simTime to see equilibrium develop
        creations[:,1] = np.random.uniform(size = nrCreations, low = 0, high = boxLength) # Locations. Needs to be adapted for multi-D
        creations[:,2:] = np.random.choice((-1,1),nrCreations)[:,np.newaxis] * np.array([1,-1]) # Creation orientation, i.e. order of +/- charges
    
    # Dependent paramaters (deducable from the ones defined above): 
    initialNrParticles = len(initialPositions)
    domain = (0,boxLength)

setExample(3)

### Create grid, e.g. as possible sources: 
nrSources = 11
sources = np.linspace(0,boxLength, nrSources)


# %%
#@jit(nopython=True)
def pairwiseDistance(x, PBCs = False):
    """ Compute distances, optionally with PBCs
    
    Computes distances between all atoms, optionally for 
    closest copies taking boundaries into account.
    
    INPUT: np array of coordinates (single or multiple dimensions)
    OUTPUT: array of difference vectors and one of distances 
            between all coordinates, optionally w/ PBCs
    """    
    
    diff = x - x[:,np.newaxis] # Difference vectors 
    if PBCs:
        diff = diff - np.floor(0.5 + diff/boxLength)*boxLength # Calculate difference vector to closest copy of particle (with correct orientation)
    dist = np.abs(diff) # Compute length of difference vectors
    
    return diff, dist



def kernel(x): 
    """ Given array of coordinates, compute log-kernel of particles """
    
    return np.log(pairwiseDistance(x))
# Idea: eventually give this as argument to `interaction' function. Currently not used


def interaction(diff,dist,b, PBCBool = True, regularisation = 'eps'):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges 
    
    Also returns array of pairwise charge products; not strictly necessary, but speeds up computation
    """
    
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to (aritrary) non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    
    if regularisation == 'eps': 
        distCorrected = (dist**2 + eps) # Normalise to avoid computational problems at singularity. Square to normalise difference vector
    else: 
        distCorrected = dist**2
    
    interactions = 1/len(b) * (-diff / distCorrected) * chargeArray #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    if regularisation == 'cutoff': 
        interactions = np.clip(interactions, -cutoff, cutoff) # Sets all values outside [-c,c] to value of closest boundary
    
    return interactions, chargeArray
#TODO: make regularisation (+eps above) variable

    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x[np.isfinite(x)] %= boxLength # Also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension
    # Index by isfinite to not affect np.nan (still works otherwise, but gives warning)

    return x



def PeachKoehler(sources, x, b, stress, regularisation = 'eps'):
    """Computes Peach-Koehler force for each source
    
    (Sources are typically fixed equidistant grid points) """
    
    x = np.nan_to_num(x)
    dist = np.abs(sources[:,np.newaxis] - x[:]) #TODO include PBCs. Normal dist function does not work here because we have two separate arrays (but may be adapted?)
    
    if regularisation == 'eps': 
        dist = (dist + eps) # Normalise to avoid computational problems at singularity #TODO doubt it is right to use same eps here!
    
    
    interactions = 2/dist + stress # According to final expression in meeting notes of 211213
    f = np.sum(interactions, axis = 1)
    
    return f
        

# %% SIMULATION


### Precomputation
nrSteps = int(simTime/dt) # Rounds fraction down to integer (floor)
if manualCrea: 
    nrCreations = 2 * len(creations) # At every creation time, two new dislocations are introduced
    nrParticles = initialNrParticles + nrCreations 
    creationSteps = np.append(np.floor(creations[:,0]/dt),0) #append 0 to 'know' last creation has happend
    creationCounter = 0
    stepsSinceCreation = np.ones(len(creations)) * -1 # Set to -1 to indicate creation has not happened yet; set to 0 at moment of creation
    exceptionSteps = int(creaExc / dt)
    b = np.append(b, np.zeros(nrCreations))
else: nrParticles = initialNrParticles


x = np.zeros((nrSteps, nrParticles))
x[0,:initialNrParticles] = initialPositions
x[0,initialNrParticles:] = np.nan # Effectively keeps particles out of system before creation

bPerTime = np.zeros((nrSteps, nrParticles))
bInitial = np.copy(b) #copy to create new array; otherwise, bInitial is just a reference (and changes if b changes)

if autoCreation: 
    tresHist = np.zeros(len(sources)) #threshold history; to measure how long the threshold was exceeded at certain source


simStartTime = timer.time() # To measure simulation computation time

### Simulation loop
for k in range(nrSteps-1):
# while t < simTime
    # Creation: 
    if autoCreation: # Idea: if force at source exceeds threshold for certain time, new dipole is created
        PK = np.abs(PeachKoehler(sources, x[k+1], b, stress))
        tresHist[PK >= forceTres] += dt # Increment all counters reaching Peach-Koehler force threshold by dt
        creations = np.where(tresHist >= timeTres)[0] # Identify which sources reached time threshold (automatically reached force threshold too)
        nrNewDislocs = len(creations) * 2
        if nrNewDislocs > 0:
            tresHist[creations] = 0 # Reset counters of new creations
            #TODO check what to do when several sources close to eachother simultaneously reach threshold
            
            #The following is a bit tedious, should be possible in a simpler way
            locs = np.zeros(nrNewDislocs)
            locs[::2] = sources[creations] - 0.5*Lnuc # Read as: every other element 
            locs[1::2] = sources[creations] + 0.5*Lnuc # Read as: every other element, starting at 1
            charges = np.tile([-1,1], len(creations)) # Creates array by repeating given pattern; this way assigns charges
            
            x = np.append(x, np.zeros((nrSteps, nrNewDislocs))*np.nan, axis = 1) #extend _entire_ position array (over all timesteps) with NaNs. #TODO can we predict a maximum a priori? May be faster than repeatedly appending
            print(np.shape(x))
            x[k, -nrNewDislocs:] = locs # replace added NaNs by creation locations for current timestep
            b = np.append(b, charges)
            bInitial = np.append(bInitial, charges)
        
    if manualCrea and (k == creationSteps[creationCounter]): 
        creationLocation = creations[creationCounter][1]
        creationOrder = creations[creationCounter][-2:]
        
        idx = initialNrParticles + 2 * creationCounter
        x[k, idx] = creationLocation - 0.5*Lnuc # Set location, distance of collision treshold apart
        x[k, idx + 1] = creationLocation + 0.5*Lnuc 
        b[idx : idx + 2] = creationOrder # Set charges as specified before
        bInitial[idx : idx + 2] = creationOrder
        
        stepsSinceCreation[creationCounter] = 0
        creationCounter += 1
    
    # main forces/interaction: 
    diff, dist = pairwiseDistance(x[k], PBCs = PBCs)
    interactions, chargeArray = interaction(diff,dist,b, PBCBool = PBCs, regularisation = reg)
    
    # Adjust forces between newly created dislocations (keeping track of time since each creation separately)
    if not autoCreation:
        #TODO do we also need this for automatic creations, or is relation w/ sheer stress etc. enough?
        for i in range(len(creations)): #TODO unnecessarily time-consuming. Should be doable without loop (or at least not every iteration)
            if (0 <= stepsSinceCreation[i] < exceptionSteps+1): # Idea: disable forces between new creation initially
                idx = initialNrParticles + 2 * i
                # Idea: make interaction forces slowly transition from -1 (opposite) to 1 (actual force)
                interactions[idx : idx + 2,idx : idx + 2,:] *= -1.0 + 2*stepsSinceCreation[i]/exceptionSteps #1 - 2/(stepsSinceCreation[i] + 1) #TODO now still assumes creations are given in order of time. May want to make more robust
                stepsSinceCreation[i] += 1
    
    updates = np.nansum(interactions,axis = 1) # Deterministic part; treating NaNs as zero
    x[k+1] = x[k] + updates * dt 
    
    if randomness: 
        random = sigma * np.random.normal(size = nrParticles) # 'noise' part
        x[k+1] += random * np.sqrt(dt) 
    
    if PBCs: 
        x[k+1] = projectParticles(x[k+1]) # Places particles back into box
    
    
    if sticky: 
        newDist = np.nan_to_num(pairwiseDistance(x[k+1], PBCs = PBCs)[1], nan = 1000) #TODO work around this, really don't want to calculate all distances twice!
        # Sets nan to arbitrary large number
        
        # Idea: dist matrix is symmetrical and we don't want the diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        newDist += 1000*np.tril(np.ones((len(x[k]),len(x[k])))) # Arbitrary large number, so that effectively all entries on and below diagonal are disregarded
        collidedPairs = np.where((newDist < collTres) & (chargeArray == -1)) # Format: ([parts A], [parts B]). Makes sure particles only annihilate if they have opposite charges
        #TODO may want something nicer than this construction... 
        
        b[collidedPairs[0]] = 0 # If non-integer charges: take (b[collidedPairs[0]] + b[collidedPairs[1]])/2
        b[collidedPairs[1]] = 0
        x[k+1, collidedPairs[0]] = np.nan # Take particles out of system
        x[k+1, collidedPairs[1]] = np.nan 
        
    # t += dt
    
    if(k % int(nrSteps/10) == 0):
        print(f"{k} / {nrSteps}")


duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %%
### Plot animation ###
def plotAnim(bInitial, x): 

    # only 0/1-values:
    if (bInitial.dtype == np.dtype('int32')):
        colorDict = {-1:'red', 0:'grey', 1:'blue'}
        cols = np.vectorize(colorDict.get)(bInitial).tolist() # Convert array of -1/0/1 to red/grey/blue
        #TODO: This way, particles that collide at some point are gray from the beginning...
        #      Can we set colours within update function?
    
    # General values of b, not only 0/1: 
    if (bInitial.dtype == np.dtype('float64')):
        n = len(bInitial)
        cols = [0]*n
        
        # Not nicest construction yet, but loop time is negligible compared to simulation
        for i in range(n): 
            if (bInitial[i] < 0): 
                cols[i] = 'red'
            # if ([i] == 0): 
            #     cols[i] = 'grey'
            if (bInitial[i] >= 0): 
                cols[i] = 'blue'
        
    
    
    fig = plt.figure()
    ax = plt.axes(xlim=domain[0], ylim=domain[1])
    scat = ax.scatter(x[0,:,0], x[0,:,1], c = cols) 
    
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    t_init = 0 # Offset e.g. when disregarding warmup period
    
    def update(i):
        update.k += 1
        xs = x[i,:,0]
        ys = x[i,:,1]
        predata = np.array([xs,ys])
        data = np.transpose(predata)
        scat.set_offsets(data)
        time_text.set_text(time_template % (t_init+i*dt))
        return scat, time_text
    
    update.k = 0
    
    ani = animation.FuncAnimation(fig, update, frames=nrSteps, interval=dt*10, )
    plt.show()


#1D plot:    
def plot1D(bInitial, x, endTime): 
    global pos, x_temp
    plt.clf() # Clears current figure
    
    trajectories = np.ndarray.squeeze(x)
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


plot1D(bInitial, x, simTime)
