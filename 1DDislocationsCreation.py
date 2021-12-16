# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:41:52 2021

@author: s161981
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:55:22 2021

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as timer
# import numba
# from numba import jit #just-in-time compiling. Yet to be implemented...

### Simulation settings
simTime = 1.5         # Total simulation time
dt = 0.002          # Timestep for discretisation
PBCs = True         # Whether to work with Periodic Boundary Conditions (i.e. see domain as torus)
eps = 0.001         # To avoid singular force makig computation instable. 
randomness = False  # Whether to add random noise to dislocation positions
sigma = 0.01        # Influence of noise
sticky = True       # Whether collisions are sticky, i.e. particles stay together once they collide
collTres = 0.008    # Collision threshold; if particles are closer than this, they are considered collided
creaExc = 0.2       # Time for which exception rule governs interaction between newly created dislocations. #TODO now still arbitrary threshold.


## Example 1:
# boxLength = 1
# initialPositions = np.array([[0.02], [0.2], [0.8], [0.85], [1]]) # Double brackets for easier generalisation to multiple dimensions
# b = np.array([-1, -1, 1, 1, -1]) # Particle charges
# # Creation time, place and 'orientation' (positive/negative charge order):
# creations = np.array([[0.14, 0.5, 1, -1],
#                       [0.3, 0.2, 1, -1],
#                       [0.6, 0.5, 1, -1],
#                       [0.61, 0.1, 1, -1]])
# Format: time, loc, orient. is (0.15, [0.5], [-1,1]). Not great, but easiest to keep overview for multiple creations
#TODO seems like two creations at exact same time are not yet possible
# Also, orientation is quite important for whether a creation immediately annihilates or not
    

### Example 2:
# boxLength = 1
# initialPositions = np.array([[0.1], [0.85]]) 
# b= np.array([-1, 1])


### Example 3:
boxLength = 1
nrParticles = 10
dim = 1
initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
# 0/1 charges:
b = np.random.choice((-1,1),nrParticles)

nrCreations = 10
creations = np.zeros((nrCreations,4))
creations[:,0] = np.linspace(0,0.5*simTime, nrCreations) # times; Last creation at 0.5 simTime to see equilibrium develop
creations[:,1] = np.random.uniform(size = nrCreations, low = 0, high = boxLength) #creaLocs
creations[:,2:] = np.random.choice((-1,1),nrCreations)[:,np.newaxis] * np.array([1,-1]) #creaOrient 



initialNrParticles = np.shape(initialPositions)[0]
dim = np.shape(initialPositions)[1]
domain = (0,boxLength)



### Create grid, e.g. as possible sources: 
# nrSourcesPerSide = 10
# nrSources = nrSourcesPerSide**2
# #TODO surely there is an easier way? (Typically meshgrid is used to plot via X,Y = meshgrid(...))
# mesh1D = np.linspace(0,boxLength, nrParticles)
# mesh2D = np.meshgrid(mesh1D, mesh1D) #creates two 2D arrays, combining elts with same indices gives point on grid
# sources = np.transpose(np.reshape(mesh2D)) #turn into list of points
# domain = ((0,boxLength),)*dim
#Define 0 to 1 as positive direction. 


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
    
    diff = x - x[:,np.newaxis] #difference vectors #2-norm of difference vectors
    if PBCs:
        diff = diff - np.floor(0.5 + diff/boxLength)*boxLength #calculate difference vector to closest copy of particle (with correct orientation)
    dist = np.linalg.norm(diff, axis = 2) #compute length of difference vectors
    
    return diff, dist



def kernel(x): 
    """ Given array of coordinates, compute log-kernel of particles """
    
    return np.log(pairwiseDistance(x))
#Idea: eventually give this as argument to `interaction' function. Currently not used


def interaction(diff,dist,b, PBCBool = True):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    np.fill_diagonal(dist, 1) # Set distance from particle to itself to non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] # Create matrix b_i b_j
    distCorrected = (dist**2 + eps) # Normalise to avoid computational problems at singularity
    interactions = 1/len(b) * (-diff / distCorrected[:,:,np.newaxis]) * chargeArray[:,:,np.newaxis]  #len(b) is nr of particles
    interactions = np.nan_to_num(interactions) # Set NaNs to 0
    
    return interactions, chargeArray
#TODO: make regularisation (+eps above) variable

    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x = x % boxLength #also works in multi-D as long as box has same length in each dimension. Else, rewrite function to modulate in each dimension

    return x



def PeachKoehler(sources, x, b):
    """Computes Peach-Koehler force for each source"""
    
    f = 0 #TODO (placeholder)
    
    return f
    #TODO correctly implement stress fields (sigma)
    #TODO correct normal vectors (probably not equal for all sources)

# %% SIMULATION


### Precomputation
nrSteps = int(simTime/dt) # Rounds fraction down to integer (floor)
nrCreations = 2 * len(creations) # At every creation time, two new dislocations are introduced
nrParticles = initialNrParticles + nrCreations 

x = np.zeros((nrSteps, nrParticles, dim))
x[0,:initialNrParticles,:] = initialPositions
x[0,initialNrParticles:,:] = np.nan # Effectively keeps particles out of system before creation

bPerTime = np.zeros((nrSteps, nrParticles))
b = np.append(b, np.zeros(nrCreations))
bInitial = np.copy(b) #copy to create new array; otherwise, bInitial is just a reference (and changes if b changes)

creationSteps = np.append(np.floor(creations[:,0]/dt),0) #append 0 to 'know' last creation has happend
creationCounter = 0
stepsSinceCreation = np.ones(len(creations)) * -1 # Set to -1 to indicate creation has not happened yet; set to 0 at moment of creation
exceptionSteps = int(creaExc / dt)



simStartTime = timer.time() #to time simulation length

### Simulation loop
for k in range(nrSteps-1):
    # Creation: 
    if (k == creationSteps[creationCounter]): 
        creationLocation = creations[creationCounter][1]
        creationOrder = creations[creationCounter][-2:]
        
        idx = initialNrParticles + 2 * creationCounter
        x[k, idx] = np.array([creationLocation - collTres]) # Set location, distance of collision treshold apart
        x[k, idx + 1] = np.array([creationLocation + collTres]) # Array construct is sort of ugly fix, but should make generalisation to 2D easier
        b[idx : idx + 2] = creationOrder # Set charges as specified before. #TODO does not seem to make a difference yet...
        
        stepsSinceCreation[creationCounter] = 0
        creationCounter += 1
    
    # main forces/interaction: 
    diff, dist = pairwiseDistance(x[k], PBCs = PBCs)
    interactions, chargeArray = interaction(diff,dist,b, PBCBool = PBCs)
    
    # Adjust forces between newly created dislocations (keeping track of time since each creation separately)
    for i in range(len(creations)): #TODO unnecessarily time-consuming. Should be doable without loop (or at least not every iteration)
        if (0 <= stepsSinceCreation[i] < exceptionSteps+1): # Idea: disable forces between new creation initially
            idx = initialNrParticles + 2 * i
            # Idea: make interaction forces slowly transition from -1 (opposite) to 1 (actual force)
            interactions[idx : idx + 2,idx : idx + 2,:] *= -1.0 + 2*stepsSinceCreation[i]/exceptionSteps #1 - 2/(stepsSinceCreation[i] + 1) #TODO now still assumes creations are given in order of time. May want to make more robust
            stepsSinceCreation[i] += 1
    
    updates = np.nansum(interactions,axis = 1) #deterministic part; treating NaNs as zero
    x[k+1] = x[k] + updates * dt 
    
    if randomness: 
        random = sigma * np.random.normal(size = (nrParticles,dim)) # 'noise' part
        x[k+1] += random * np.sqrt(dt) 
    
    if PBCs: 
        x[k+1] = projectParticles(x[k+1]) #places particles back into box
    
    
    if sticky: 
        newDist = np.nan_to_num(pairwiseDistance(x[k+1], PBCs = PBCs)[1], nan = 1000) #TODO work around this, really don't want to calculate all distances twice!
        #set nan to arbitrary large number
        
        # Idea: dist matrix symmetrical and don't want diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        newDist += 1000*np.tril(np.ones((nrParticles,nrParticles))) #arbitrary large number, so that effectiv
        collidedPairs = np.where((newDist < collTres) & (chargeArray == -1)) #format ([parts A], [parts B]). Makes sure particles only annihilate if they have opposite charges
        #TODO may want something nicer than this construction... 
        
        b[collidedPairs[0]] = 0 #(b[collidedPairs[0]] + b[collidedPairs[1]])/2
        b[collidedPairs[1]] = 0
        x[k+1, collidedPairs[0]] = np.nan #(x[k+1, collidedPairs[0]] + x[k+1, collidedPairs[1]])/2 
        x[k+1, collidedPairs[1]] = np.nan #(x[k+1, collidedPairs[0]] + x[k+1, collidedPairs[1]])/2 
        
        # bPerTime[collidedPairs[0]] = b[collidedPairs[0]] #to keep track of colours in animated plot (not yet working)
        # bPerTime[collidedPairs[1]] = b[collidedPairs[1]]
        
    
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
    if (bInitial.dtype == dtype('int32')):
        colorDict = {-1:'red', 0:'grey', 1:'blue'}
        cols = np.vectorize(colorDict.get)(bInitial).tolist() #convert array of -1/0/1 to red/grey/blue
        #TODO: this way, particles that collide at some point are gray from the beginning...
        #      can we set colours within update function?
    
    # general values of b, not only 0/1: 
    if (bInitial.dtype == dtype('float64')):
        n = len(bInitial)
        cols = [0]*n
        
        #not nicest construction yet, but loop time is negligible compared to simulation
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
    t_init = 0 # offset e.g. when disregarding warmup period
    
    N = nrParticles
    
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
def plot1D(bInitial, x, endTime, nrCreations): 
    global pos, x_temp
    plt.clf() # Clears current figure
    
    trajectories = np.ndarray.squeeze(x)
    colorDict = {-1:'red', 0:'grey', 1:'blue'}
    bInitial[-nrCreations:] = creations[:,-2:].flatten() 
    #set colour of created dislocations according to charge they eventually get (not 0 as beginning)
    
    nrPoints, nrTrajectories = np.shape(trajectories)
    
    plt.ylim((0,endTime))
    y = np.linspace(0, endTime, nrPoints)
    
    for i in range(nrTrajectories):
        #insert NaNs at 'discontinuities':
        x_current = trajectories[:,i]
        x_temp = np.nan_to_num(x_current, nan = 10**5) # Work-around to avoid invalid values in np.where below
        pos = np.where(np.abs(np.diff(x_temp)) >= 0.5)[0]+1
        x_new = np.insert(x_current, pos, np.nan)
        y_new = np.insert(y, pos, np.nan) #insert NaNs in order not to draw jumps across domain caused by PBCs
        #note that x[pos] = np.nan would work as well, but that would delete data
        
        plt.plot(x_new, y_new, c = colorDict.get(bInitial[i]))

#TODO: figure out how to plot trajectories w/ particles jumping sides.


plot1D(bInitial, x, simTime, nrCreations)
