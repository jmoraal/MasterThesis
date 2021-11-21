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


boxLength = 7
domain = ((0,boxLength),(0,boxLength))
#Define 0 to 1 as positive direction. 

# Example 1:
# initialPositions = np.array([[0.02,0.02], [0.2,0.8], [0.8,0.5], [0.85,0.3], [1,0.8]]) 
# charges = np.array([-1,1,1,-1,-1])
# nrParticles = np.shape(initialPositions)[0]
# dim = np.shape(initialPositions)[1]


# Example 2:
nrParticles = 20
dim = 2
initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
charges = np.random.choice((-1,1),nrParticles)
# charges = np.ones(nrParticles)



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


def interaction(x,charges, PBCBool = True):
    """ Compute array of pairwise particle interactions for given array of particle coordinates and charges """
    
    diff, dist = pairwiseDistance(x, PBCs = PBCBool)
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



### Simulation ###

nrSteps = 500
dt = 0.01
PBCs = True # Periodic Boundary Conditions
eps = 0.00001
sigma = 0.03 # influence of noise

x = np.zeros((nrSteps, nrParticles, dim))
x[0] = initialPositions
simStartTime = timer.time() #to time simulation length

for k in range(nrSteps-1):
    determ = np.sum(interaction(x[k],charges, PBCBool = PBCs),axis = 1) #deterministic part
    random = sigma * np.random.normal(size = (nrParticles,dim)) #'noise' part
    x[k+1] = x[k] + determ * dt +  random * np.sqrt(dt) 
    x = projectParticles(x) #places particles back into box
    
    if(k % (nrSteps/10) == 0):
        print(f"{k} / {nrSteps}")

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', 
                                  int((duration%3600)/60), " minutes and ", 
                                  int(duration%60), "seconds")  



### Plot animation ###

colorDict = {-1:'red', 1:'blue'}
cols = np.vectorize(colorDict.get)(charges).tolist() #convert array of -1/1 to red/blue



fig = plt.figure()
ax = plt.axes(xlim=domain[0], ylim=domain[1])
scat = ax.scatter(x[0,:,0], x[0,:,1], c = cols) 


time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
t_init = 0 # offset e.g. when disregarding warmup period

N = nrParticles

def update(i):
    xs = x[i,:,0]
    ys = x[i,:,1]
    predata = np.array([xs,ys])
    data = np.transpose(predata)
    scat.set_offsets(data)
    time_text.set_text(time_template % (t_init+i*dt))
    return scat, time_text


ani = animation.FuncAnimation(fig, update, frames=nrSteps, interval=dt*10)
plt.show()
