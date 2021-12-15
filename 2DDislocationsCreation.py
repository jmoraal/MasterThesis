# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:55:22 2021

@author: s161981
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time as timer
import numba
from numba import jit #just-in-time compiling. Yet to be implemented...



# Example 1:
# boxLength = 1
# initialPositions = np.array([[0.02,0.02], [0.2,0.8], [0.8,0.5], [0.85,0.3], [1,0.8]]) 
# b = np.array([-1,1,1,-1,-1])
# nrParticles = np.shape(initialPositions)[0]
# dim = np.shape(initialPositions)[1]

# Example 2:
boxLength = 1
nrParticles = 50
dim = 2
initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
# 0/1 charges:
b = np.random.choice((-1,1),nrParticles)
# arbitrary charges in [-1,1]: 
b = b * np.random.rand(nrParticles)
# b = np.ones(nrParticles)
nrSourcesPerSide = 10
nrSources = nrSourcesPerSide**2
#TODO surely there is an easier way? (Typically meshgrid is used to plot via X,Y = meshgrid(...))
mesh1D = np.linspace(0,boxLength, nrParticles)
mesh2D = np.meshgrid(mesh1D, mesh1D) #creates two 2D arrays, combining elts with same indices gives point on grid
sources = np.transpose(np.reshape(mesh2D)) #turn into list of points

domain = ((0,boxLength),)*dim
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
    interactions = 1/nrParticles * (-diff / distCorrected[:,:,np.newaxis]) * chargeArray[:,:,np.newaxis] 
    
    return interactions
#TODO: make regularisation (+eps above) variable!

    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x = x % boxLength #also works in multi-D as long as box has same length in each dimension 

    return x



def PeachKoehler(sources, x, b):
    """Computes Peach-Koehler force for each source"""
    
    f = 0 #TODO (placeholder)
    
    return f
    #TODO correctly implement stress fields (sigma)
    #TODO correct normal vectors (probably not equal for all sources)

# %%
### Simulation ###

nrSteps = 500
dt = 0.003
PBCs = True # Periodic Boundary Conditions
randomness = False
eps = 0.00001
sigma = 0.01 # influence of noise
sticky = True # whether or not collisions are sticky
collTres = 0.01

x = np.zeros((nrSteps, nrParticles, dim))
bPerTime = np.zeros((nrSteps, nrParticles))
x[0] = initialPositions
simStartTime = timer.time() #to time simulation length
bInitial = b
for k in range(nrSteps-1):
    diff, dist = pairwiseDistance(x[k], PBCs = PBCs)
    
    if sticky: 
        # Idea: dist matrix symmetrical and don't want diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        dist += 10*np.tril(np.ones((nrParticles,nrParticles))) #arbitrary large number, so that effectiv
        collidedPairs = np.where(dist < collTres) #format ([parts A], [parts B])
        #TODO may want something nicer than this construction... 
        
        b[collidedPairs[0]] = (b[collidedPairs[0]] + b[collidedPairs[1]])/2
        b[collidedPairs[1]] = 0
        x[k, collidedPairs[1]] = (x[k, collidedPairs[0]] + x[k, collidedPairs[0]])/2 
        #TODO this is not precise enough; make sure only net charge 0 can annihilate
        
        # bPerTime[collidedPairs[0]] = b[collidedPairs[0]] #to keep track of colours in animated plot (not yet working)
        # bPerTime[collidedPairs[1]] = b[collidedPairs[1]]
        
    determ = np.sum(interaction(diff,dist,b, PBCBool = PBCs),axis = 1) #deterministic part
    x[k+1] = x[k] + determ * dt 
    
    if randomness: 
        random = sigma * np.random.normal(size = (nrParticles,dim)) #'noise' part
        x[k+1] += random * np.sqrt(dt) 
    
    
    
    x = projectParticles(x) #places particles back into box
    
    if(k % (nrSteps/10) == 0):
        print(f"{k} / {nrSteps}")

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  


# %%
### Plot animation ###
#if only 0/1-values:
# colorDict = {-1:'red', 0:'grey', 1:'blue'}
# cols = np.vectorize(colorDict.get)(b).tolist() #convert array of -1/0/1 to red/grey/blue
#TODO: this way, particles that collide at some point are gray from the beginning...
#      can we set colours within update function?

#if general values of b, not only 0/1: 
cols = [0]*nrParticles

for i in range(nrParticles): 
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
