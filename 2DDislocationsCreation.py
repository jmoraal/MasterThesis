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
# charges = np.array([-1,1,1,-1,-1])
# nrParticles = np.shape(initialPositions)[0]
# dim = np.shape(initialPositions)[1]

# Example 2:
boxLength = 1
nrParticles = 50
dim = 2
initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
charges = np.random.choice((-1,1),nrParticles)
# charges = np.ones(nrParticles)

domain = ((0,boxLength),(0,boxLength))
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


def interaction(diff,dist,charges, PBCBool = True):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    np.fill_diagonal(dist, 1) #set distance from particle to itself to non-zero value to avoid div by 0; arbitrary, since this term cancels out anyway
    chargeArray = charges * charges[:,np.newaxis] #create matrix b_i b_j
    distCorrected = (dist**2 + eps) #normalise to avoid computational problems at singularity
    interactions = 1/nrParticles * (-diff / distCorrected[:,:,np.newaxis]) * chargeArray[:,:,np.newaxis] 
    
    return interactions
#TODO: make regularisation (+eps above) variable!

    
def projectParticles(x):
    """Projects particles into box of given size."""
    
    x = x % boxLength #also works in multi-D as long as box has same length in each dimensions 

    return x



# def PeachKoehler(x, charges):
# May not be possible in this IPS-like system...

# %%
### Simulation ###

nrSteps = 500
dt = 0.001
PBCs = True # Periodic Boundary Conditions
randomness = False
eps = 0.00001
sigma = 0.01 # influence of noise
sticky = True # whether or not collisions are sticky
collTres = 0.01

x = np.zeros((nrSteps, nrParticles, dim))
chargesPerTime = np.zeros((nrSteps, nrParticles))
x[0] = initialPositions
simStartTime = timer.time() #to time simulation length

for k in range(nrSteps-1):
    diff, dist = pairwiseDistance(x[k], PBCs = PBCs)
    
    if sticky: 
        # Idea: dist matrix symmetrical and don't want diagonal, so only take entries above diagonal. 
        #       need to compare to threshold, so np.triu does not work since all others are set to 0
        #       so set everything on and below diagonal to arbitrary large value
        dist += 10*np.tril(np.ones((nrParticles,nrParticles))) #arbitrary large number, so that effectiv
        collidedPairs = np.where(dist < collTres) #format ([parts A], [parts B])
        #TODO may want something nicer than this construction... 
        
        charges[collidedPairs[0]] = 0
        charges[collidedPairs[1]] = 0
        chargesPerTime[collidedPairs[0]] = 0 #to keep track of colours in animated plot (not yet working)
        x[k, collidedPairs[1]] = x[k, collidedPairs[0]] #TODO Better: set to avg of both
        
        #TODO this is not precise enough; make sure only net charge 0 can annihilate
        
    determ = np.sum(interaction(diff,dist,charges, PBCBool = PBCs),axis = 1) #deterministic part
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

colorDict = {-1:'red', 0:'grey', 1:'blue'}
cols = np.vectorize(colorDict.get)(charges).tolist() #convert array of -1/0/1 to red/grey/blue
#TODO: this way, particles that collide at some point are gray from the beginning...
#      can we set colours within update function?


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
