import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#%%
H0=100*0.67*10**(3)/(3.086*10**(22))
HI=1e-6
If = 1
pi = np.pi

def PB(t):
    k=1
    res = 9*HI**4/(8*pi**2)*(1+(k*t)**2 /3+(k*t)**4/9)
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
    res= HI**4*m*t**4*(81 + 45 * k**2*t**2 + 22*k**4*t**4+k**6*t**6)/(8*k**2*pi**2)
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
#%%
# Now 1G ~ HI*1e-40 M_pl**2
#Planck mass M_pl = 2.176434*1e-8

GC = 1e-40 #Gauss convert
HIG=1e46
afa0 = 1e-29

delta_B=np.sqrt((9*HI**4)/(4*pi**2*If**2))

delta_B0= np.sqrt((9*HI**4)/(4*pi**2*If**2))*afa0**2
plusPI1 = 1e-6/(afa0**2*1e46)

PI0 = plusPI1-1
#%%
def PiBtest(k):
    res = 1/9*(9+3*k**2+k**4)
    return res

kappa_vals = np.linspace(0.01, 1e3, 10000)  
pbs = PiBtest(kappa_vals)

plt.loglog(kappa_vals, pbs, 'teal',label=r"$experimental \Pi_B$")
plt.ylabel(r"$\Pi_B$")
plt.xlabel(r"$\kappa$")
plt.legend()
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)

plt.show()
#%%
def PiEtest(k):
    res = 1 +1/k**2+2/3 *k**2+k**4/9
    return res

pes=PiEtest(kappa_vals)
plt.loglog(kappa_vals, pes, 'teal',label=r"$experimental \Pi_E$")
plt.ylabel(r"$\Pi_E$")
plt.xlabel(r"$\kappa$")
plt.legend()
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7)

plt.show()
#%%
