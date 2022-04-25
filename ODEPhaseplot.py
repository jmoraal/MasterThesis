# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:57:01 2022

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import describe
from scipy.optimize import fsolve
from scipy.optimize import root


#TODO definition of gamma depends on texc! Is this correctly adjusted? 

F = 1
texc = 0.5
rmin = 0
rmax = 1.0
tmax = 2.5


#%% (Adapted) Creation ODE and 'creation exception function' gamma: 
def gamma(t, texc = texc):
    return min(2*t/texc - 1, 1)
    # if t < texc: 
    #     return 2*t/texc - 1
    # else: 
    #     return 1


def f(t, R, gamma = gamma, F = F, texc = texc): 
    #TODO no abs! (now fixed later, but not very neat.)
    return -gamma(t, texc = texc) + F*np.sqrt(2*np.abs(R)) # np.abs(R) yields non-physical negative solutions for R (idea of starting at negative distance). To be accounted for later on!



#%% Plots: 

### Plot inherent boundaries and system constants
def plotFeatures(): 
    # global thres
    
    plt.clf()
    # thres = 1/(2*F**2)
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
def fprime(t, R, texc = texc): 
    return -f(texc - t, R, texc = texc)


def critical(F, texc = texc, plot = False): 
    global critR, thres
    
    thres = np.reshape(1/(2*F**2), (1,)) # Reshape is technicality, to account both for case where F is given as 'c' or '[c]'
    
    boundSol2 = solve_ivp(fprime, [0, texc], thres, method = 'BDF', max_step = 5e-2) 
    critR = boundSol2.y[0][-1]
    
    if plot: 
        boundSol1 = solve_ivp(f, [texc, tmax], [np.squeeze(thres)], method = 'BDF', max_step = 5e-2) 
        plt.plot(boundSol1.t,np.squeeze(boundSol1.y), color = 'black')
        plt.plot(texc - boundSol2.t,np.squeeze(boundSol2.y), color = 'black') # 'invert' ODE to solve backwards to t=0
        
        print(f"Critical initial condition: R = {critR}" )
    
    return critR



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
    critical(F, plot = True)
    plotSols()
    plotPhaseVF()


# plotAll()

#%% More critical-value analysis: critical initial value for R, varying F and texc


def initialCrit(Fmin, Fmax, N, tmin, tmax, M, plot = True): 
    plt.clf() # Clears current figure
    
    Fs = np.linspace(Fmin, Fmax, N)
    ts = np.linspace(tmin, tmax, M)
    crits = np.zeros((M,N)) #not strictly  necessary to store all, but may come in handy
    for j in range(M): 
        texc = ts[j]
        for i in range(N): 
            crits[j,i] = critical(Fs[i], texc = texc)
        
        Rs = crits[j,:] 
        Rs[Rs < 0] = np.nan #remove negative values (only there so solver works more easily)
        plt.plot(Fs, Rs, label = "$t_{exc}$ =" + f" {ts[j]:.2f}") #Note: only round (:.2f) in case linspace is coarse!
    
    plt.hlines(0, 0, Fmax)
    plt.xlabel('$F$')
    plt.ylabel('$R_{crit}$')
    plt.xlim([Fmin, Fmax])
    
    plt.legend()
    
    # describe(crits)

# initialCrit(Fmin, Fmax, N, tmin, tmax, M)


#%% One level deeper: for each texc, pick F s.t. Rcrit = 0. (bad computation time...)


def FzeroSolver(tmin, tmax, M, method):
    global texc
    
    ts = np.linspace(tmin, tmax, M)
    Fzeros = np.zeros(M)
    reachedSol = np.zeros(M, dtype = int)
    
    for j in range(M): 
        texc = ts[j]
        if method == 'fsolve': 
            x, infodict, ier, msg = fsolve(critical, 0.5/texc, full_output = True) # use 'args = ...' kwarg?
            Fzeros[j] = x
            # print(x, critical(x))
            
            reachedSol[j] = (ier == 1)
                
        if method == 'root': 
            sol = root(critical, 0.5/texc, method = 'broyden1')
            Fzeros[j] = sol.x
            reachedSol[j] = sol.success
            
        print(f"Step {j} out of {M}")
    
    #TODO may be possible to do this much faster by avoiding for-loop; requires adapting 'critical' to also take vector input (and output)
    
    plt.clf()
    tsProper = ts[reachedSol == 1]
    FzerosProper = Fzeros[reachedSol == 1]
    plt.plot(tsProper, FzerosProper) #Note: only round (:.2f) in ca
    
    # plt.hlines(0, 0, Fmax)
    plt.xlabel('$t_{exc}$')
    plt.ylabel('$F$')
    # plt.xlim([Fmin, Fmax])




#%% Executables: 

# plotAll() # Phase-space, trajectories & vector field


Fmin = 0.2
Fmax = 10
N = 100

tmin = 0.2
tmax = 1
M = 5
initialCrit(Fmin, Fmax, N, tmin, tmax, M) # Critical R for varying F, several texc


# tmin = 0.2
# tmax = 0.6
# M = 100
# # method = 'root'  #very slow, but seems accurate
# method = 'fsolve' # fast but error-prone
# FzeroSolver(tmin, tmax, M, method) # Relate F and texc to critical R belonging to R(0) = 0

