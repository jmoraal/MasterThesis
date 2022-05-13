# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:57:01 2022

@author: s161981
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import curve_fit


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
    # np.abs(R) yields non-physical negative solutions for R (idea of starting 
    # at negative distance); prevents errors. This is accounted for later on
    return -gamma(t, texc = texc) + F*np.sqrt(2*np.abs(R)) 



#%% Plots: 

### Plot inherent boundaries and system constants
def plotFeatures(): 
    
    plt.clf()
    plt.vlines(texc, rmin, 10*rmax, linestyles = 'dotted') #vertical line at t = texc
    
    #"level set" of points with derivative 0
    def separation(t):
        return 0.5*((gamma(t))/F)**2
    
    ts = np.linspace(0.5*texc, tmax, 100)
    ys = np.zeros(100)
    
    for i in range(100): 
        ys[i] = separation(ts[i])



def fprime(t, R, texc = texc): 
    """ plots the 'critical solution', above which R 
    diverges and below which R goes to 0: """
    return -f(texc - t, R, texc = texc)


def critical(F, texc, plot = False): 
    """ given F, texc, finds initial condition yielding equilibrium solution """
    global critR, thres
    
    thres = np.reshape(1/(2*F**2), (1,)) 
    # (Reshape is technicality, to account both for F given as 'c' or '[c]')
    
    boundSol2 = solve_ivp(fprime, [0, texc], thres, 
                          method = 'BDF', max_step = 5e-2) 
    critR = boundSol2.y[0][-1]
    
    if plot: 
        boundSol1 = solve_ivp(f, [texc, tmax], [np.squeeze(thres)], 
                              method = 'BDF', max_step = 5e-2) 
        plt.plot(boundSol1.t,np.squeeze(boundSol1.y), color = 'black')
        # 'invert' ODE because we solve backwards to t=0:
        plt.plot(texc - boundSol2.t,np.squeeze(boundSol2.y), color = 'black') 
        
        print(f"Critical initial condition: R = {critR}" )
    
    return critR



def plotSols():
    """Print range of solution trajectories """
    for R0 in np.array([0, 0.5, 0.9, 1.1, 1.5, 2, 3])*critR:
        sol = solve_ivp(f, [0,tmax], [R0], method = 'BDF', max_step = 5e-2) 
        #set maximum timestep, else solver 'gets arrogant' and inaccurate
        plt.plot(sol.t,np.squeeze(sol.y))



def plotPhaseVF():
    """ Plot phase-space vector field"""
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
    """ Execute all plot-functions above """
    plotFeatures()
    critical(F, texc, plot = True)
    plotSols()
    plotPhaseVF()



#%% More critical-value analysis: critical initial value for R, varying F and texc


def initialCrit(Fmin, Fmax, N, tmin, tmax, M, plot = True): 
    """ For range of forces and times, plots IC yielding equilibrium solution """
    plt.clf() # Clears current figure
    
    Fs = np.linspace(Fmin, Fmax, N)
    ts = np.linspace(tmin, tmax, M)
    crits = np.zeros((M,N)) 
    for j in range(M): 
        texc = ts[j]
        for i in range(N): 
            crits[j,i] = critical(Fs[i], texc)
        
        Rs = crits[j,:] 
        Rs[Rs < 0] = np.nan #remove negative values (so solver works more easily)
        plt.plot(Fs, Rs, label = "$t_{exc}$ =" + f" {ts[j]:.2f}") 
        # (only round (:.2f) in case linspace is coarse)
    
    plt.hlines(0, 0, Fmax)
    plt.xlabel('$F$')
    plt.ylabel('$R_{crit}$')
    plt.xlim([Fmin, Fmax])
    
    plt.legend()
    
    # describe(crits)


def plotRcrit(): 
    Fmin = 0.2
    Fmax = 10
    N = 100
    
    tmin = 0.2
    tmax = 1
    M = 5
    initialCrit(Fmin, Fmax, N, tmin, tmax, M) # Critical R for varying F, texc


#%% One level deeper: for each texc, pick F s.t. Rcrit = 0. (bad computation time...)


def FzeroSolver(ts, method):
    """ For each texc, attempts to find F s.t. Rcrit = 0 """
    
    M = len(ts)
    Fzeros = np.zeros(M)
    reachedSol = np.zeros(M, dtype = int)
    
    for j in range(M): 
        texc = ts[j]
        if method == 'fsolve': 
            x, infodict, ier, msg = fsolve(critical, 1/texc, 
                                           full_output = True, args = (texc,)) 
            Fzeros[j] = x
            
            reachedSol[j] = (ier == 1)
                
        if method == 'root': 
            sol = root(critical, 1/texc, method = 'broyden1', args = texc)
            Fzeros[j] = sol.x
            reachedSol[j] = sol.success # Store whether root-finder was succesful
            
        print(f"Step {j} out of {M}, succesful: {reachedSol[j]}")
    
        
    plt.clf()
    tsProper = ts[reachedSol == 1]
    FzerosProper = Fzeros[reachedSol == 1]
    plt.plot(tsProper, FzerosProper) 
    
    plt.xlabel('$t_{exc}$')
    plt.ylabel('$F$')
    
    return tsProper, FzerosProper



def plotRcritZeros():
    tmin = 0.35
    tmax = 1.5
    M = 100
    ts = np.linspace(tmin, tmax, M) #evenly distributed
    method = 'root'  #very slow, but seems accurate
    # method = 'fsolve' # fast but error-prone
    
    # For given t, find F s.t. Rcrit = 0: 
    tsProper, FzerosProper = FzeroSolver(ts, method) 
    
    return tsProper, FzerosProper


def modelFunc(x, d,e,f,k): 
    """ Function to fit to FzeroSolver """
    
    return d/(x-k) + e/(x-k)**2 + f/(x-k)**3


def fitFunction(F,t): 
    """ Fit and plot approximation of FzeroSolver """
    
    # Fit data to modelFunc: 
    params, _ = curve_fit(modelFunc, t, F, p0 = (1,1,1,0.3)) # p0 is first guess
    
    
    print("Optimal parameters: ", params)
    a,b,c,d = params
    print(f"Optimal function: f(x) = {a}/(x-{d}) + {b}/(x-{d})**2 + {c}/(x-{d})**3")
    print(f"Rounded: f(x) = {a:.3f}/(x-{d:.3f}) \
          + {b:.3f}/(x-{d:.3f})**2 + {c:.3f}/(x-{d:.3f})**3")
    
    # Plot original and fitted function: 
    tSol = np.linspace(0.37, 1.5, 1000)
    FSol = modelFunc(tSol, a,b,c,d)
    
    plt.clf()
    plt.plot(FSol, tSol, label = "Fitted", color = 'orange', zorder = 0)
    plt.plot(F, t, label="Original", zorder = 1)
    plt.xlabel('$F$')
    plt.ylabel('$t_{exc}$')
    plt.legend()



#%% Executables (reproducing report): 

## Plot phase-space and range of solutions: 
plotAll() # Phase-space, trajectories & vector field


## Plot Rcrit:
# plotRcrit()


## Find and plot zeros of Rcrit: 
t, F = plotRcritZeros()


## Fit solution to curve of given form:  [see below arrays]

# Solving is slow, so may take following data instead of plotRcritZeros(): 
    
F = np.array([4.52249233,  3.25458218,  2.66708927,  2.30421261,
        2.05243906,  1.86080379,  1.71050999,  1.58773009,  1.48501998,
        1.39732073,  1.32130856,  1.25497266,  1.19522794,  1.14207018,
        1.09396342,  1.05035073,  1.01065947,  0.97400351,  0.94019942,
        0.90895822,  0.87986711,  0.8529585 ,  0.82745218,  0.80383456,
        0.78152787,  0.76045845,  0.74060595,  0.72184348,  0.70408308,
        0.68725822,  0.67118846,  0.65595032,  0.64140218,  0.62749767,
        0.61426562,  0.60162896,  0.5894524 ,  0.5778138 ,  0.56663221,
        0.55590293,  0.54561666,  0.53566809,  0.52610706,  0.51692251,
        0.50805281,  0.49946465,  0.49118272,  0.48316996,  0.47545857,
        0.46797684,  0.46075236,  0.45371297,  0.4469083 ,  0.44031849,
        0.43393784,  0.42771814,  0.42166954,  0.4158048 ,  0.41011023,
        0.40458888,  0.39918341,  0.39394864,  0.38883055,  0.38386148,
        0.37903058,  0.37429789,  0.36969419,  0.36520798,  0.36084195,
        0.35656104,  0.35238137,  0.34830612,  0.34432858,  0.34045044,
        0.33664585,  0.33292527,  0.32929026,  0.32575529,  0.32227046,
        0.31886116,  0.31553138,  0.31227159,  0.30909019,  0.30595748,
        0.30288957,  0.29988414,  0.29694334,  0.29406616,  0.29123311,
        0.28845638,  0.28573354,  0.28307541,  0.28045133,  0.27787487,
        0.27534974])

t = np.array([0.37323232, 0.38484848, 0.39646465, 0.40808081,
        0.41969697, 0.43131313, 0.44292929, 0.45454545, 0.46616162,
        0.47777778, 0.48939394, 0.5010101 , 0.51262626, 0.52424242,
        0.53585859, 0.54747475, 0.55909091, 0.57070707, 0.58232323,
        0.59393939, 0.60555556, 0.61717172, 0.62878788, 0.64040404,
        0.6520202 , 0.66363636, 0.67525253, 0.68686869, 0.69848485,
        0.71010101, 0.72171717, 0.73333333, 0.74494949, 0.75656566,
        0.76818182, 0.77979798, 0.79141414, 0.8030303 , 0.81464646,
        0.82626263, 0.83787879, 0.84949495, 0.86111111, 0.87272727,
        0.88434343, 0.8959596 , 0.90757576, 0.91919192, 0.93080808,
        0.94242424, 0.9540404 , 0.96565657, 0.97727273, 0.98888889,
        1.00050505, 1.01212121, 1.02373737, 1.03535354, 1.0469697 ,
        1.05858586, 1.07020202, 1.08181818, 1.09343434, 1.10505051,
        1.11666667, 1.12828283, 1.13989899, 1.15151515, 1.16313131,
        1.17474747, 1.18636364, 1.1979798 , 1.20959596, 1.22121212,
        1.23282828, 1.24444444, 1.25606061, 1.26767677, 1.27929293,
        1.29090909, 1.30252525, 1.31414141, 1.32575758, 1.33737374,
        1.3489899 , 1.36060606, 1.37222222, 1.38383838, 1.39545455,
        1.40707071, 1.41868687, 1.43030303, 1.44191919, 1.45353535,
        1.46515152])


# fitFunction(F,t)
