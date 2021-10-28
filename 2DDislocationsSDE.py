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
from numba import jit #just-in-time compiling


boxLength = 7
domain = ((0,boxLength),(0,boxLength))

#Define 0 to 1 as positive direction. 
# Example 1:
# initialPositions = np.array([[0.02,0.02], [0.2,0.8], [0.8,0.5], [0.85,0.3], [1,0.8]]) # double brackets for easier generalisation to multiple dimensions
# charges = np.array([-1,1,1,-1,-1])
# nrParticles = np.shape(initialPositions)[0]
# dim = np.shape(initialPositions)[1]


# Example 2:
nrParticles = 10
dim = 2
initialPositions = np.random.uniform(size = (nrParticles, dim), low = 0, high = boxLength)
charges = np.random.choice((-1,1),nrParticles)
# charges = np.ones(nrParticles)



#@jit(nopython=True)
def pairwiseDistance(x, PBCs = False):
    """ Computes distances between all atoms, optionally in closest copies taking boundaries into account."""    
    
    diff = x - x[:,np.newaxis] #difference vectors #2-norm of difference vectors
    if PBCs:
        diff = diff - np.floor(0.5 + diff/boxLength)*boxLength #calculate difference vector to closest copy of particle (with correct orientation)
    dist = np.linalg.norm(diff, axis = 2)
    
    return diff, dist



def kernel(x): 
    return np.log(pairwiseDistance(x))


# def energy(x,charges, kernel = kernel):
#     E = 0
#     K = kernel(x) #n*n matrix
#     for i in range(nrParticles):
#         for j in range(nrParticles):
#             if (i != j):
#                 E += charges[i]*charges[j]*K[i][j]/nrParticles


def interaction(x,charges, PBCBool = True):
    diff, dist = pairwiseDistance(x, PBCs = PBCBool)
    np.fill_diagonal(dist, 1) #something going wrong here?
    chargeArray = charges * charges[:,np.newaxis] #create matrix b_i b_j
    distCorrected = (dist**2 + eps)
    temp = 1/nrParticles * (-diff / distCorrected[:,:,np.newaxis]) * chargeArray[:,:,np.newaxis] 
    
    return temp
    
    
def projectParticles(x):
    """Projects particles into box of given size."""
    x = x % boxLength #also works in multi-D as long as box has same dimensions in every direction

    return x

### Simulation ###

nrSteps = 500
dt = 0.001
PBCs = True
eps = 0.00001
noiseFactor = 0.03

x = np.zeros((nrSteps, nrParticles, dim))
x[0] = initialPositions
simStartTime = timer.time() #to time simulation length

for k in range(nrSteps-1):
    determ = np.sum(interaction(x[k],charges, PBCBool = PBCs),axis = 1) #deterministic part
    random = noiseFactor * np.sqrt(dt) * np.random.normal(size = (nrParticles,dim)) #'noise' part
    x[k+1] = x[k] + determ * dt + np.sqrt(dt) * random
    x = projectParticles(x) #places particles back into box
    
    if(k % (nrSteps/10) == 0):
        print(f"{k} / {nrSteps}")

duration = timer.time() - simStartTime
print("Simulation duration was ", int(duration/3600), 'hours, ', int((duration%3600)/60), " minutes and ", int(duration%60), "seconds")  



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
    #predata = np.array([x[i][:N], x[i][N:]])
    xs = x[i,:,0]
    ys = x[i,:,1]
    predata = np.array([xs,ys])
    data = np.transpose(predata)
    scat.set_offsets(data)
    time_text.set_text(time_template % (t_init+i*dt))
    return scat, time_text


ani = animation.FuncAnimation(fig, update, frames=nrSteps, interval=dt*10)
plt.show()