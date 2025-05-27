import matplotlib.pyplot as plt
import numpy as np
# from scipy.integrate import quad, dblquad
import jax.numpy as jnp
import jax
from jax import jit, vmap, lax
from scipy.stats import qmc
from scipy import constants
import time
#%%
HI = 1e-6
TR=-0.01
pi = np.pi
alpha = 1e7
Delta_tau = 1e-1
kstar = 1000
mpl = constants.physical_constants["Planck mass"][0]
aR = -1/(HI*TR)
#%%
start = time.time()

@jit
def C0(t,s):
    t1num = (s**2+t*(t+2)-1)**2
    t1denom = 4*(-s+t+1)**2
    
    t2num = (s**2+t*(t+2)+3)**2
    t2denom = 4*(s+t+1)**2
    
    t1 = (t1num/t1denom)+1
    t2 = (t2num/t2denom)+1

    return t1*t2

# @jit
# def C0(t,s):
#     t1num = (s**2+t*(t+2)+3)**2*(s**2+t*(t+2)-1)**2
#     t1denom = 16*(-s+t+1)**2*(s+t+1)**2
#     # if t1denom<1e-4:
#     #     t1 = 0
#     # else:
#     #     t1 = t1num/t1denom
#     t2num = (s**2+t*(t+2)-1)**2
#     t2denom = 2*(-s+t+1)**2
#     # if t2denom<1e-4:
#     #     t2 = 0
#     # else:
#     #     t2 = t2num/t2denom
#     t1 = t1num/t1denom
#     t2 = t2num/t2denom
#     return 1 + t1 + t2

def Si(x):    
    num_samples = 1000
    sampler = qmc.Sobol(d=1, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [0]
    u_bounds = [x]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    f = lambda xb: jnp.sin(xb)/xb
    intvals = vmap(f)(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    # jax.debug.print("SI:{}", integral)
    return integral


# I use the definition for Ci from wikipedia where it's: gamma+ln x-Cin(x). Where Cin(x)=int 0-> x (1-cost)/t dt
def Ci(x):
    num_samples = 1000
    sampler = qmc.Sobol(d=1, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [0]
    u_bounds = [x]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    f = lambda xb: - (1-jnp.cos(xb))/xb
    intvals = vmap(f)(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    
    # jax.debug.print("CI:{}", integral)
    return jnp.euler_gamma + jnp.log(x) +integral


@jit
def PB(k):
    pb = 1 +2/k**3 * Pi0*(k**3+(4*k**2-3)*jnp.sin(2*k)-(k**2-6)*k*jnp.cos(2*k))+4/k**6*Pi0**2*(k**2+1)*((k**2-3)*jnp.sin(k)+3*k*jnp.cos(k))**2
    res = lax.cond(k>=50, lambda _:0.0, lambda _: pb, operand = None)
    return res/(1+0.01*k**8)
                                                       


@jit
def IuvPB238(args, k):
    s, t = args
    f = lambda s, t: 1/(1-s+t)**2*1/(1+s+t)**2*PB(k*((t+s+1)/2))*PB(k*((t-s+1)/2))*C0(t,s)
    return 1/8*f(s,t)



def OmegaGW_qmc(k, num_samples=800000):
    sampler = qmc.Sobol(d=2, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [-1, 0]
    u_bounds = [1, 30]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    intvals = vmap(lambda args: IuvPB238(args, k))(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    
    # t1 = pi/384 * ((9*HI**4)/(4*pi**2*mpl**4))*HI**4*TR**8*mpl**4
    t2 = (Ci(k*xstar)**2+(pi/2+Si(k*xstar))**2)
    # jax.debug.print("t2:{}", t2)

    return t2*integral
Pi0 = 1e6
xstar = 1e-3

k_vals = jnp.logspace(-5,1.7,1000)
# k_vals = jnp.linspace(1e-2,1e4,1000)
# Omeg = vmap(OmegaGW_qmc)(k_vals)
Omeg = jnp.abs(jnp.array(list(map(OmegaGW_qmc, k_vals))))
end = time.time()
print("Calculation Time:", end-start)
#%%
plt.loglog(k_vals, Omeg)


# plt.xlim(1e-3,10)
plt.xlabel(r"$\kappa$", size = 14)
# plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.title(r"QMC Sobol: kstar = 1000, $\Pi_0$ = 1e6, $x_\star$=1e-3")
# plt.ylim(1e5, 1e25)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/SobolOmegGW_newc0.png', bbox_inches='tight')

plt.show()

#%%
from scipy.optimize import curve_fit

def power_law(kappa, A, B):
    # return A * kappa**B
    return  np.log(A)+B*np.log(kappa)
# Select the range where you want to fit the power law
fit_range = (k_vals > 3e-2) & (k_vals < 1e-1)

kappa_fit = k_vals[fit_range]
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
plt.loglog(k_vals, Omeg, label=r'Original $f(\kappa)$', color='blue', alpha=0.5)
# Plot the selected data for fitting
plt.scatter(kappa_fit, np.exp(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.plot(kappa_fit, np.exp(fitted_curve), label=r'Power-Law Fit: $A \kappa^B$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}')
plt.legend()
plt.grid()
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits1.png', bbox_inches='tight')

plt.show()
#%%
def power_law(kappa, A, B):
    # return A * kappa**B
    return  np.log(A)+B*np.log(kappa)

# Select the range where you want to fit the power law
fit_range = (k_vals > 2e-1) & (k_vals < 2e1)

kappa_fit = k_vals[fit_range]
Omeg_fit = np.log(Omeg[fit_range])

# Perform power-law fitting
popt, pcov = curve_fit(power_law, kappa_fit, Omeg_fit)

# Extract fitted parameters
A_fit, B_fit = popt
print(f"Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve = power_law(kappa_fit, A_fit, B_fit)

# Plot the original function
plt.loglog(k_vals, Omeg, label=r'Original $f(\kappa)$', color='blue', alpha=0.5)
# Plot the selected data for fitting
plt.scatter(kappa_fit, np.exp(Omeg_fit), color='black', label='Data for Fit')
# Plot the power-law fit
plt.plot(kappa_fit, np.exp(fitted_curve), label=r'Power-Law Fit: $A \kappa^B$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.title(f'Fitted Power-Law: A = {A_fit:.4e}, B = {B_fit:.4f}')
plt.legend()
plt.grid()
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/Powerlaw fits2.png', bbox_inches='tight')

plt.show()

