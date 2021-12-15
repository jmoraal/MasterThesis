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



### Example 1:
boxLength = 1
initialPositions = np.array([[0.02], [0.2], [0.8], [0.85], [1]]) # double brackets for easier generalisation to multiple dimensions
b = np.array([-1, -1, 1, 1, -1]) 

# Creation time, place and 'orientation' (positive/negative charge order):
creation = (0.3, [0.5], [-1,1])

    
### Example 1:
boxLength = 1
# initialPositions = np.array([[0.25], [0.75]]) 
# b= np.array([-1, 1])


initialNrParticles = np.shape(initialPositions)[0]
dim = np.shape(initialPositions)[1]
domain = (0,boxLength)


### Example 2:
# boxLength = 1
# nrParticles = 6
# dim = 1
# initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
# # 0/1 charges:
# b = np.random.choice((-1,1),nrParticles)
# # arbitrary charges in [-1,1]: 
# b = b * np.random.rand(nrParticles)
# # b = np.ones(nrParticles)


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
#Idea: eventually give this as argument to `interaction' function


def interaction(diff,dist,b, PBCBool = True):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    np.fill_diagonal(dist, 1) #set distance from particle to itself to non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = b * b[:,np.newaxis] #create matrix b_i b_j
    distCorrected = (dist**2 + eps) #normalise to avoid computational problems at singularity
    interactions = 1/len(b) * (-diff / distCorrected[:,:,np.newaxis]) * chargeArray[:,:,np.newaxis]  #len(b) is nr of particles
    
    return interactions
#TODO: make regularisation (+eps above) variable!

    
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

### Simulation settings
simTime = 1
dt = 0.003
PBCs = False # Periodic Boundary Conditions
randomness = False
eps = 0.001 # to avoid singular force makig computation instable. 
sigma = 0.01 # influence of noise
sticky = True # whether or not collisions are sticky
collTres = 0.001 #Collision threshold; if particles are closer than this, they are considered collided
#TODO: additionally require particles to have opposite charges (otherwise, there
#      exist initial configurations where same-charge particles wrongly collide)


### Precomputation
nrSteps = int(simTime/dt) #rounds fraction down to integer (floor)
nrCreations = 2 #*len(creation[0]) #at every creation time, two new dislocations are introduced
nrParticles = initialNrParticles + nrCreations

x = np.zeros((nrSteps, nrParticles, dim))
bPerTime = np.zeros((nrSteps, nrParticles))
x[0,:initialNrParticles,:] = initialPositions
simStartTime = timer.time() #to time simulation length
b = np.append(b, np.zeros(nrCreations))
bInitial = np.copy(b) #copy to create new array; otherwise, bInitial is just a reference (and changes if b changes)

creationStep = np.floor(creation[0]/dt)


### Simulation loop
for k in range(nrSteps-1):
    # Creation: 
    if (k == creationStep): 
        creationLocation, creationOrder = creation[1], creation[2]
        x[k, -2] = creationLocation + np.array([creationOrder[0] * collTres * 0.5]) # set location, distance of collision treshold apart
        x[k, -1] = creationLocation + np.array([creationOrder[1] * collTres * 0.5]) # sort of ugly fix, but should make generalisation to 2D easier
        b[-2:] = creationOrder #set charges as specified before
    
    # main forces/interaction: 
    diff, dist = pairwiseDistance(x[k], PBCs = PBCs)
    interactions = interaction(diff,dist,b, PBCBool = PBCs)
    
    if (k - creationStep < 10): #TODO now still arbitrary threshold. Idea: disable forces between new creation initially
        interactions[:,-2:,-2:] = 0 
    
    updates = np.sum(interactions,axis = 1) #deterministic part
    x[k+1] = x[k] + updates * dt 
    
    if randomness: 
        random = sigma * np.random.normal(size = (nrParticles,dim)) #'noise' part
        x[k+1] += random * np.sqrt(dt) 
    
    if PBCs: 
        x[k+1] = projectParticles(x[k+1]) #places particles back into box
    
    newDist = pairwiseDistance(x[k+1], PBCs = PBCs)[1] #TODO work around this, really don't want to calculate all distances twice!
    
    if sticky: 
        # Idea: dist matrix symmetrical and don't want diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        newDist += 10*np.tril(np.ones((nrParticles,nrParticles))) #arbitrary large number, so that effectiv
        collidedPairs = np.where(newDist < collTres) #format ([parts A], [parts B])
        #TODO may want something nicer than this construction... 
        
        b[collidedPairs[0]] = 0 #(b[collidedPairs[0]] + b[collidedPairs[1]])/2
        b[collidedPairs[1]] = 0
        x[k+1, collidedPairs[0]] = x[k+1, collidedPairs[1]]  #(x[k+1, collidedPairs[0]] + x[k+1, collidedPairs[1]])/2 
        # x[k+1, collidedPairs[1]] = (x[k+1, collidedPairs[0]] + x[k+1, collidedPairs[1]])/2 
        #TODO this is not precise enough; make sure only net charge 0 can annihilate
        
        # bPerTime[collidedPairs[0]] = b[collidedPairs[0]] #to keep track of colours in animated plot (not yet working)
        # bPerTime[collidedPairs[1]] = b[collidedPairs[1]]
        
        
        
    
    
    
    if(k % (nrSteps/10) == 0):
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
def plot1D(bInitial, x): 
    plt.clf() # Clears current figure
    
    trajectories = np.ndarray.squeeze(x)
    colorDict = {-1:'red', 0:'grey', 1:'blue'}
    # cols = np.vectorize(colorDict.get)(bInitial).tolist()
    # plt.gca().set_color_cycle(cols)
    # plt.plot(trajectories, range(nrSteps)) #c = cols does not work; only takes one colour at a time
    
    nrPoints, nrTrajectories = np.shape(trajectories)
    
    for i in range(nrTrajectories):
        plt.plot(trajectories[:,i], range(nrPoints), c = colorDict.get(bInitial[i]))

#TODO: figure out how to plot trajectories w/ particles jumping sides.

plot1D(bInitial, x)