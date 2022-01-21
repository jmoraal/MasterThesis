# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:57:01 2022

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


F = 1
texc = 0.5
rmin = 0
rmax = 1.0
tmax = 2.5


### (Adapted) Creation ODE and 'creation exception function' gamma: 
def gamma(t, texc = texc):
    return min(2*t/texc - 1, 1)
    # if t < texc: 
    #     return 2*t/texc - 1
    # else: 
    #     return 1


def f(t, R, gamma = gamma, F = F): 
    return -gamma(t) + F*np.sqrt(2*np.abs(R))



    
### Plot inherent boundaries
def plotFeatures(): 
    global thres
    
    plt.clf()
    thres = 1/(2*F**2)
    # plt.hlines(thres, 0.5, tmax, linestyles = 'dashed')
    plt.vlines(texc, rmin, 10*rmax, linestyles = 'dotted') #vertical line at t = texc
    
    #"level set" of derivative-0 pts
    def separation(t):
        return 0.5*((gamma(t))/F)**2
    
    ts = np.linspace(0.5*texc, tmax, 100)
    ys = np.zeros(100)
    
    for i in range(100): 
        ys[i] = separation(ts[i])
        
    # plt.plot(ts, ys, color = 'black', linestyle = 'dashed')


## The following plots the 'critical solution', above which R diverges and below which R goes to 0: 
def fprime(t, R): 
    return -f(texc - t, R)

def plotCritical(): 
    global critR
    
    boundSol1 = solve_ivp(f, [texc, tmax], [thres], method = 'BDF', max_step = 5e-2) 
    boundSol2 = solve_ivp(fprime, [0, texc], [thres], method = 'BDF', max_step = 5e-2) 
    plt.plot(boundSol1.t,np.squeeze(boundSol1.y), color = 'black')
    plt.plot(texc - boundSol2.t,np.squeeze(boundSol2.y), color = 'black') # 'invert' ODE to solve backwards to t=0
    
    critR = boundSol2.y[0][-1]
    print(f"Critical initial condition: R = {critR}" )



### Print some other solution trajectories: 
def plotSols():
    for R0 in np.array([0, 0.5, 0.9, 1.1, 1.5, 2, 3])*critR:
        sol = solve_ivp(f, [0,tmax], [R0], method = 'BDF', max_step = 5e-2) #set maximum timestep, else solution is not smooth (inaccurately so)
        plt.plot(sol.t,np.squeeze(sol.y))



### Plot phase-space vector field: 
def plotPhaseVF():
    rlim = max(rmax, 2.5*thres)
    ts = np.linspace(0, tmax, 20)
    rs = np.linspace(0, rlim, 20)
    T, R = np.meshgrid(ts, rs)
    
    
    u, v = np.zeros(T.shape), np.zeros(R.shape)
    NI, NJ = T.shape
    
    for i in range(NI):
        for j in range(NJ):
            der = f(T[i, j],R[i, j]) #derivative value according to ODE
            u[i,j] = T[i,j]
            v[i,j] = der
         
    
    
    plt.quiver(T, R, u, v, alpha = 0.5) # in form (x coords, y coords, x dir, y dir)
    
    plt.xlabel('$t$')
    plt.ylabel('$R$')
    plt.ylim([rmin, rmax])
    plt.xlim([0,tmax])


def plotAll(): 
    plotFeatures()
    plotCritical()
    plotSols()
    plotPhaseVF()


