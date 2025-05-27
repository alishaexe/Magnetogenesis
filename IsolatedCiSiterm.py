import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici
#%%
pi = np.pi
xstar = 1e-3

def Si(x):
    return sici(x)[0]

def Ci(x):
    return sici(x)[1]


def term(k):
    t1 = k**2*(Ci(k*xstar)**2+(pi/2+Si(k*xstar))**2)
    return t1

k = np.logspace(-2,4,1000)

termsol = np.array(list(map(term, k)))
#%%
plt.loglog(k, termsol)
plt.xlabel(r"$\kappa$", size = 14)
plt.ylabel(r"Term", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Scipy version")
# plt.ylim(top = 10**5)
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.png', bbox_inches='tight')

plt.show()