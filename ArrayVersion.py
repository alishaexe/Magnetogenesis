import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.special import sici
# %%
HI = 1e-6
TR=-0.01
pi = np.pi
alpha = 1e7
Delta_tau = 1e-1
Pi0 = 1e6
kstar = 1000
# %%


def PB238(k):
    res = 1 +2/k**3 * Pi0*(k**3+(4*k**2-3)*np.sin(2*k)-(k**2-6)*k*np.cos(2*k))+4/k**6*Pi0**2*(k**2+1)*((k**2-3)*np.sin(k)+3*k*np.cos(k))**2
    return res/(1+0.01*k**8) 

def C0(t,s):
    t1num = (s**2+t*(t+2)+3)**2*(s**2+t*(t+2)-1)**2
    t1denom = 16*(-s+t+1)**2*(s+t+1)**2
    
    t2num = (s**2+t*(t+2)-1)**2
    t2denom = 2*(-s+t+1)**2
    
    t1 = t1num/t1denom
    t2 = t2num/t2denom
    
    return 1 + t1 + t2