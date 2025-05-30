import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.special import sici
from scipy import constants
from scipy.optimize import curve_fit

# %%
HI = 1e-6
TR=-0.01
pi = np.pi
alpha = 1e7
Delta_tau = 1e-1
kstar = 1000
mpl = constants.physical_constants["Planck mass"][0]
aR = -1/(HI*TR)
Pi0 = 1e6
xstar = 1e-3
# %%
start = time.time()
def Si(x):
    res = sici(x)[0]
    return res

def Ci(x):
    res = sici(x)[1]
    return res

def PB(k):
    if k>=50:
        return 0
    else:
        res = 1 +2/k**3 * Pi0*(k**3+(4*k**2-3)*np.sin(2*k)-(k**2-6)*k*np.cos(2*k))+4/k**6*Pi0**2*(k**2+1)*((k**2-3)*np.sin(k)+3*k*np.cos(k))**2
        return res

def C0(t,s):
    t1num = (s**2+t*(t+2)-1)**2
    t1denom = 4*(-s+t+1)**2
    
    t2num = (s**2+t*(t+2)+3)**2
    t2denom = 4*(s+t+1)**2
    
    t1 = (t1num/t1denom)+1
    t2 = (t2num/t2denom)+1

    return t1*t2

kT = np.logspace(-5, 3, 800)
tT = np.logspace(-2, 2.2, 200)
sT = np.linspace(-1, 1, 75)


i = range(len(tT))
j = range(len(sT))
m = range(len(kT))

coords = np.array(np.meshgrid(i, j)).T.reshape(-1,2)

def sub(m,i,j):
    
    res = (-C0(tT[i], sT[j]))*PB((tT[i]+sT[j]+1)/2*kT[m])*PB((tT[i]-sT[j]+1)/2*kT[m])
    return res

def OGW(m):
    test = np.array(list(map(lambda args: sub(m,*args)*tT[args[0]]*sT[args[1]], coords)))
    pb = np.sum(test)
    t1 = pi/384 * ((9*HI**4)/(4*pi**2*mpl**4))**2*(HI**4*TR**8*mpl**4)/96
    t2 = (Ci(kT[m]*xstar)**2+(pi/2+Si(kT[m]*xstar))**2)
    return t1*t2*pb


Omeg = np.array(list(map(OGW,m)))
end = time.time()
print("Calculation Time:", end-start)

#%%
plt.loglog(kT, Omeg)
plt.xlabel(r"$\kappa$", size = 14)
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Array_OmegaGW.png', bbox_inches='tight')

plt.show()
#%%
def power_law(kappa, A, B):
    # return A * kappa**B
    return  np.log(A)+B*np.log(kappa)

fit_range = (kT > 3e-4) & (kT < 3e-2)

kappa_fit = kT[fit_range]
Omeg_fit = np.log(Omeg[fit_range])

#Needed to do Log since it was such a small range/kappa too small and Omeg too big


# Perform power-law fitting
popt, pcov = curve_fit(power_law, kappa_fit, Omeg_fit)

# Extract fitted parameters
A_fit, B_fit = popt
print(f"Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve = power_law(kappa_fit, A_fit, B_fit)

# Plot the original function
plt.loglog(kT, Omeg, label=r'Original $f(\kappa)$', color='blue', alpha=0.5)
# Plot the selected data for fitting
plt.scatter(kappa_fit, np.exp(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.loglog(kappa_fit, np.exp(fitted_curve), label=r'Power-Law Fit: $A \kappa^B$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits1.png', bbox_inches='tight')

plt.show()
#%%
def power_law(kappa, A, B):
    # return A * kappa**B
    return  np.log(A)+B*np.log(kappa)
# Select the range where you want to fit the power law
fit_range = (kT > 1e-1) & (kT < 4e1)

kappa_fit = kT[fit_range]
Omeg_fit = np.log(Omeg[fit_range])

# Perform power-law fitting
popt, _ = curve_fit(power_law, kappa_fit, Omeg_fit)

# Extract fitted parameters
A_fit, B_fit = popt
print(f"Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve = power_law(kappa_fit, A_fit, B_fit)

# Plot the original function
plt.loglog(kT, Omeg, label=r'Original $f(\kappa)$', color='blue', alpha=0.5)
# Plot the selected data for fitting
plt.scatter(kappa_fit, np.exp(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.loglog(kappa_fit, np.exp(fitted_curve), label=r'Power-Law Fit: $A \kappa^B$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits2.png', bbox_inches='tight')

plt.show()