import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.special import sici
from scipy import constants
from scipy.optimize import curve_fit

# %%
pi = np.pi

# kstar = 1000
Mpl = constants.physical_constants["Planck mass"][0]
HI = Mpl*1e-6 #This ensures HI/Mpl = 1e-6
Pi0 = 7e7
xstar = 1e-4
omegrd = 6.6e-5*0.68**(-2)
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

# def C0(t,s):
#     t1num = (s**2+t*(t+2)-1)**2
#     t1denom = 4*(-s+t+1)**2
    
#     t2num = (s**2+t*(t+2)+3)**2
#     t2denom = 4*(s+t+1)**2
    
#     t1 = (t1num/t1denom)+1
#     t2 = (t2num/t2denom)+1

#     return t1*t2

def C0(t,s):
    u = (t+s+1)/2
    v = (t-s+1)/2
    mu = (1-u**2-v**2)/(2*v)
    
    res = (1+mu**2)*(1+(1-mu*v)**2/u**2)
    return res

kT = np.logspace(np.log10(7e-3), np.log10(90), 800)
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
    t1 = ((9*HI**4)/(32*pi**2*Mpl**4))**2 * 1/(3*omegrd)
    t2 = (Ci(kT[m]*xstar)**2+(pi/2-Si(kT[m]*xstar))**2)
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
    return  np.log10(A)+B*np.log10(kappa)

fit_range = (kT > 1e-2) & (kT < 3e-2)

kappa_fit1 = kT[fit_range]
Omeg_fit1 = np.log10(Omeg[fit_range])

#Needed to do Log since it was such a small range/kappa too small and Omeg too big


# Perform power-law fitting
popt, pcov = curve_fit(power_law, kappa_fit1, Omeg_fit1)

# Extract fitted parameters
A_fit, B_fit = popt
print(f"Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve1 = power_law(kappa_fit1, A_fit, B_fit)

# Plot the original function
plt.loglog(kT, Omeg, label=r'Original $f(\kappa)$', color='black')
# Plot the selected data for fitting
# plt.scatter(kappa_fit, 10**(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.loglog(kappa_fit1, 10**(fitted_curve1),'--', label=r'Power-Law Fit: $\log_{10} A + B\log_{10}\kappa$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits1.png', bbox_inches='tight')

plt.show()
#%%
def power_law(kappa, A, B):
    # return A * kappa**B
    return  np.log10(A)+B*np.log10(kappa)
# Select the range where you want to fit the power law
fit_range = (kT > 1e-1) & (kT < 4e1)

kappa_fit2 = kT[fit_range]
Omeg_fit2 = np.log10(Omeg[fit_range])

# Perform power-law fitting
popt, _ = curve_fit(power_law, kappa_fit2, Omeg_fit2)

# Extract fitted parameters
A_fit2, B_fit2 = popt
print(f"Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve2 = power_law(kappa_fit2, A_fit2, B_fit2)

# Plot the original function
plt.loglog(kT, Omeg, label=r'Original $f(\kappa)$', color='black')
# Plot the selected data for fitting
# plt.scatter(kappa_fit, 10**(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.loglog(kappa_fit2, 10**(fitted_curve2),'--', label=r'Power-Law Fit: $\log_{10} A + B\log_{10}\kappa$', color='blue')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit2:.4e}, B = {B_fit2:.4f}')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits2.png', bbox_inches='tight')

plt.show()
#%%
# Plot the original function
plt.loglog(kT, Omeg, label=r'$\Omega_{GW}(\kappa)$', color='black')
# plt.loglog(kappa_fit1, 10**(fitted_curve1),'--', label=r'Power-Law Fit: $f^{B}$'.format(B=B_fit:.4f), color='red')
plt.loglog(kappa_fit2, 10**(fitted_curve2),'--', label=fr'Power-Law Fit: $f^{{{B_fit2:.4f}}}$', color='blue')
plt.loglog(kappa_fit1, 10**(fitted_curve1),'--', label=fr'Power-Law Fit: $f^{{{B_fit:.4f}}}$', color='red')

plt.xlabel(r'$f/f_\star$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits.png', bbox_inches='tight')

plt.show()

#%%
bpl = np.load('/Users/alisha/Documents/LISA_ET/Datafiles/BPLS/BPLS_lisa.npy')
bpl = bpl[bpl[:, 1] < 1e-5]

plt.loglog(kT, Omeg, label=r'$\Omega_{GW}(\kappa)$', color='black')
plt.loglog((bpl[:,0])/1e-1, (bpl[:,1]), label = "LISA BPLS curve", color = "lime", linewidth=2.5)
plt.xlabel(r'$f/f_\star$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.xlim(1e-5,20)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Comp with BPL.png', bbox_inches='tight')

plt.show()