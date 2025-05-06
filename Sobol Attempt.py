import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
import jax.numpy as jnp
import jax
from jax import jit, vmap
from scipy.stats import qmc
#%%
HI = 1e-6
TR=-1e6
pi = jnp.pi
alpha = 1.05E7
Delta_tau = 0.0013
Pi0 = 10
kstar = 1000
#%%
@jit
def C0(t,s):
    term1 = s**4*(t*(t+2)+2)**2
    term2 = 2*s**2*(t-2)*(t*(t+4)+2)
    term3 = (t*(t+2)+2)**2
    term4 = (s**2-(t+1)**2)**2
    return 2*(term1+term2+term3)/term4


def Si(x):    
    num_samples = 100
    sampler = qmc.Sobol(d=1, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [0]
    u_bounds = [x]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    f = lambda xb: jnp.sin(xb)/xb
    intvals = vmap(f)(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = -jnp.mean(intvals) * volume
    # jax.debug.print("SI:{}", integral)
    return integral


#So originally the intgeral is x -> infinity HOWEVER
#It doesnt like it if you have x>200 even though there's a - sign infront of the integral
#THINK ABOUT THIS
def Ci(x):
    num_samples = 100
    sampler = qmc.Sobol(d=1, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [200]
    u_bounds = [x]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    f = lambda xb: jnp.cos(xb)/xb
    intvals = vmap(f)(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    
    # jax.debug.print("CI:{}", integral)
    return integral

@jit
def PB238(k):
    kappa = k/kstar
    term1 = 1+ 2*Pi0*(kappa**3+(4*kappa**2-3)*jnp.sin(2*kappa)-(kappa*2-6)*kappa*jnp.cos(2*kappa))/kappa**3
    term2 = (4*(kappa**2+1)*Pi0**2*((kappa**2-3)*jnp.sin(kappa)+3*kappa*jnp.cos(kappa))**2)/kappa**6
    return (term1+term2)* (9*HI**4)/(4*pi**2)

@jit
def IuvPB238_qmc(args, k):
    s, t = args
    f = (1-s+t)*(1+s+t)*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)
    jax.debug.print("Iuv:{}", f)
    return 1/8*f


def OmegaGW_qmc(k, num_samples=300000):
    sampler = qmc.Sobol(d=2, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [-1, 0]
    u_bounds = [1, 200]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    intvals = vmap(lambda args: IuvPB238_qmc(args, k))(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    
    t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = Ci(-k*TR)**2+(pi/2+Si(-k*TR))**2
    return t1*t2*integral

k_vals = jnp.logspace(0,3,10)
# Omeg = vmap(OmegaGW_qmc)(k_vals)
Omeg = jnp.array(list(map(OmegaGW_qmc, k_vals)))

#%%
plt.loglog(k_vals, Omeg)