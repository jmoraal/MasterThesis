# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:01:07 2021

@author: s161981
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

nrSteps = 20
dt = 1
PBCs = False
domain = (0,1)
#Define 0 to 1 as positive direction. 
#TODO: re-think how to set domain. Probably best if it always starts at 0, so need not specify that here
# initialPositions = np.array([[0.02], [0.2], [0.8], [0.85], [1]]) # double brackets for easier generalisation to multiple dimensions
# initialPositions = np.array([[0], [0.25], [0.5], [0.75], [1]]) 
# charges = np.array([-1, -1, 1, 1, -1]) # or does 0/1 make more sense?

initialPositions = np.array([[0.25], [0.75]]) 
charges = np.array([-1, 1])

nrParticles = np.shape(initialPositions)[0]
dim = np.shape(initialPositions)[1]
masses = np.ones(nrParticles)
v0 = 0.1*np.random.uniform(size=dim*nrParticles).reshape((nrParticles,dim)) #first generates long 1D-array, then reshapes to desired dimensions
v0 = v0 * 0 #or set initial velocity to 0



def pairwiseDistance(x, PBCs = False):
    """ Computes distances between all atoms, optionally in closest copies taking boundaries into account."""    
    
    diff = x - x[:,np.newaxis] #difference vectors #2-norm of difference vectors
    if PBCs:
        diff = diff - np.floor(0.5 + diff/domain[1])*domain[1]
    dist = np.linalg.norm(diff, axis = 2)
    
    return diff, dist

def projectParticles(x):
    """Projects particles into box of given size."""
    x = x % domain[1] #also works in multi-D as long as box has same dimensions in every direction

    return x

def integratorEuler(x, v, a, dt):
    """ Implementation of a single step for Euler integrator.""" 
    
    x = x + dt*v + (dt**2)/2*a
    v = v + dt*a
    return(x, v)


# def integratorVerlocity(x, v, a, dt):  
#     """ Implementation of a single step for Velocity Verlet integrator.""" 
#     x_new = x + v*dt + (dt**2)/2*a
#     forces = force(x_new)
#     a_new = forces/m[:,np.newaxis]
#     v = v + dt/2*(a_new + a)
#     return(x_new, v, a_new)
# Problem: requires force computation inside integrator function. Not well generalisable
# Alternative: give force function as argument. Might become messy

def integratorVerlocity1(x, v, a, dt):  
    """ Implementation of the first part of a single step for Velocity Verlet integrator.""" 
    x_new = x + v*dt + (dt**2)/2*a
    
    return x_new

def integratorVerlocity2(x_new, v, a_new, dt):  
    """ Implementation of the first part of a single step for Velocity Verlet integrator.""" 
    v_new = v + dt/2*(a_new + a)
    
    return x_new, v_new, a_new

def computeForces(x, charges, PBCs = False):
    global forces, diff, dist, pairs, distReciprocal, forceMagnitudes, directedForces, normalisedDiff, pairDists
    '''Returns magnitude of interacting force of two particles'''
    forces = np.zeros((nrParticles, dim), dtype = float)
    
    diff,dist = pairwiseDistance(x, PBCs = PBCs)
    
    pairs = np.where(np.triu(dist, 1) > 0) #tuple of arrays; sort of adjacency list. 
    #triu to avoid duplicates, 1 to start from 1st upper diagonal to avoid self-paris (leading to division by 0)
    #on second thought, may not be necessary. Does >0 work with rounding errors?
    #At least allows for easy cut-off for particles too far apart
    
    forceMagnitudes = 0.002 * dist[pairs[0],pairs[1]] ** -2
    #important formula! Look for realistic computation. (-2 ipv -1 to normalise for difference vector length)
    
    directedForces = forceMagnitudes[:,np.newaxis] * diff[pairs[0],pairs[1]] 
    signs = charges[pairs[0]] * charges[pairs[1]] 
    #elementwise multiplication of charges ensures same signs repel while opposite signs attract
    
    directedForces = directedForces * signs[:,np.newaxis]
    
    np.add.at(forces, pairs[0], -directedForces) 
    np.add.at(forces, pairs[1], directedForces) #Newton; force on one particle is opposite the other
    
    return forces


### Simulation ###
x = initialPositions
v = v0    
trajectories = np.zeros((nrSteps, nrParticles, dim), dtype = float)
for i in range(nrSteps):
    forces = computeForces(x, charges, PBCs = PBCs)
    a = forces/masses[:,np.newaxis]
    # x, v = integratorEuler(x, v, a, dt) #instable, as results show. 
    x_new = integratorVerlocity1(x, v, a, dt)
    forces = computeForces(x_new, charges, PBCs = PBCs) #TODO store this to re-use; not necessary to calculate forces twice
    a_new = forces/masses[:,np.newaxis]
    x, v, a = integratorVerlocity2(x_new, v, a_new, dt)
    
    if PBCs: 
        x = projectParticles(x)
    else: 
        v[np.where(x < domain[0])] = 0
        v[np.where(x > domain[1])] = 0
        x[np.where(x < domain[0])] = domain[0]
        x[np.where(x > domain[1])] = domain[1]
        #yet to be adapted to multi-d
    
    trajectories[i] = x
    
    #TODO: somehow recognise dislocations having met and let them continue as one
    

### Analysis ###
#1D plot:    
trajectories = np.ndarray.squeeze(trajectories)

plt.clf() # Clears current figure
plt.plot(trajectories, range(nrSteps))

#TODO: figure out how to plot trajectories w/ particles jumping sides.
