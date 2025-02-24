import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#%%
H0=100*0.67*10**(3)/(3.086*10**(22))
pi = np.pi

def PB(t):
    k=1
    res = 9*H0**4/(8*pi**2)*(1+(k*t)**2 /3+(k*t)**4/9)
    return res

# kt_vals = np.linspace(0, 100, 100)  
t_vals = np.linspace(0, 100, 10000) 
power_spectraB = PB(t_vals)

#%%
plt.loglog(t_vals, power_spectraB, 'purple',label=r"$\mathcal{P}_B$")
plt.ylabel(r"$\mathcal{P}_B$")
plt.xlabel(r"$\tau$")
plt.legend()
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/MagSpec.png', bbox_inches='tight')

plt.show()
#%%
def PE(t):
    m=1
    k=1
    res= H0**4*m*t**4*(81 + 45 * k**2*t**2 + 22*k**4*t**4+k**6*t**6)/(8*k**2*pi**2)
    return res
t_vals = np.linspace(0, 100, 10000) 
power_spectraE = PE(t_vals)
#%%
plt.loglog(t_vals, power_spectraE, 'teal',label=r"$\mathcal{P}_E$")
plt.ylabel(r"$\mathcal{P}_E$")
plt.xlabel(r"$\tau$")
plt.legend()
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/ElecSpec.png', bbox_inches='tight')

plt.show()