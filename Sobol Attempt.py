import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
import jax.numpy as jnp
import jax
from jax import jit, vmap
from scipy.stats import qmc
#%%
HI = 1e-6
TR=-0.01
pi = np.pi
alpha = 1e7
Delta_tau = 1e-1
kstar = 1000
#%%
# @jit
# def C0(t,s):
#     term1 = s**4*(t*(t+2)+2)**2
#     term2 = 2*s**2*(t-2)*(t*(t+4)+2)
#     term3 = (t*(t+2)+2)**2
#     term4 = (s**2-(t+1)**2)**2
#     return 2*(term1+term2+term3)/term4
@jit
def C0(t,s):
    t1 = s**4*(t*(t+2)+2)**2
    t2 = 4*s**3*t*(t+1)*(t+2)
    t3 = 2*s**2*(t-2)*(t*(t+4)+2)
    tdenom = (s**2-(t+1)**2)**2
    t4 = -(4*s*t*(t+1)*(t+2)+(t*(t+2)+2)**2)
    return (t1+t2+t3)/tdenom+t4/tdenom

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


# I use the definition for Ci from wikipedia where it's: gamma+ln x-Cin(x). Where Cin(x)=int 0-> x (1-cost)/t dt
def Ci(x):
    num_samples = 100
    sampler = qmc.Sobol(d=1, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [0]
    u_bounds = [x]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    f = lambda xb: jnp.euler_gamma + jnp.log(x) - (1-jnp.cos(xb))/xb
    intvals = vmap(f)(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    
    # jax.debug.print("CI:{}", integral)
    return integral

# def Si(x):
#     integrand = lambda xb: np.sin(xb)/xb
#     res = quad(integrand, 0, x)[0]
#     return res

# def Ci(x):
#     integrand = lambda xb: np.cos(xb)/xb
#     res = -quad(integrand, x, np.inf)[0]
#     return res
@jit
def PB(k):
    kappa = k/kstar
    denominator = 64 * (-1 + Delta_tau)**8 * kappa**6
    
    term1 = jnp.exp(2j * kappa) * alpha * (-3j + 2 * (-1 + Delta_tau) * kappa)
    term2 = jnp.exp(4j * Delta_tau * kappa) * alpha * (-3j * (1 + 4 * Delta_tau) + 
            2 * (2 + (7 - 3 * Delta_tau) * Delta_tau) * kappa -
            2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa**2)
    term3 = jnp.exp(2j * Delta_tau * kappa) * alpha * (3j + 2 * (-1 + Delta_tau) * kappa * 
            (2 - 1j * (-1 + Delta_tau) * kappa))
    term4 = jnp.exp(2j * (1 + Delta_tau) * kappa) * (
            8 * (-1 + Delta_tau)**4 * kappa**3 + alpha * (3j + 2 * kappa +
            2 * Delta_tau * (6j + kappa * (2 - 9 * Delta_tau - 
            2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa + 
            2 * (-1 + Delta_tau)**2 * (1 + Delta_tau) * kappa**2))))
    
    numerator1 = (jnp.exp(1j * (-2 + 3 * Delta_tau) * kappa) * alpha * (3j + 2 * (-1 + Delta_tau) * kappa) +
                  (alpha * (3j * (1 + 4 * Delta_tau) + 2 * (2 + (7 - 3 * Delta_tau) * Delta_tau) * kappa +
                  2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa**2)) / jnp.exp(1j * Delta_tau * kappa) +
                  jnp.exp(1j * Delta_tau * kappa) * alpha * (-3j + 2 * (-1 + Delta_tau) * kappa *
                  (2 + 1j * (-1 + Delta_tau) * kappa)) +
                  jnp.exp(1j * (-2 + Delta_tau) * kappa) * (
                      8 * (-1 + Delta_tau)**4 * kappa**3 + alpha * (-3j + 2 * kappa +
                      2 * Delta_tau * (-6j + kappa * (2 - 9 * Delta_tau +
                      2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa +
                      2 * (-1 + Delta_tau)**2 * (1 + Delta_tau) * kappa**2)))))
    
    numerator = (term1 + term2 + term3 + term4) * numerator1 / jnp.exp(3j * Delta_tau * kappa)
    return jnp.real(numerator / denominator * (9*HI**4)/(4*pi**2))/kappa**3


@jit
def PB238(k):
    kappa = k/kstar
    term1 = 1+ 2*Pi0*(kappa**3+(4*kappa**2-3)*jnp.sin(2*kappa)-(kappa*2-6)*kappa*jnp.cos(2*kappa))/kappa**3
    term2 = (4*(kappa**2+1)*Pi0**2*((kappa**2-3)*jnp.sin(kappa)+3*kappa*jnp.cos(kappa))**2)/kappa**6
    
    # jax.debug.print("t1+t2:{}", (9*HI**4)/(4*pi**2))
    return (term1+term2)*(9*HI**4)/(4*pi**2)

@jit
def IuvPB238_qmc(args, k):
    s, t = args
    # f = (1-s+t)*(1+s+t)*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)
    f = (1-s+t)*(1+s+t)*PB(k*((t+s+1)/2))*PB(k*((t-s+1)/2))*C0(t,s)

    # jax.debug.print("PBt+s:{}", 1e30*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s))
    # jax.lax.cond(
    # jnp.any(jnp.isnan(jnp.array([PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)]))),
    # lambda _: jax.debug.print("NaN detected in PB238(t+s)!"),
    # lambda _: None,
    # operand=None
    # )
    return 1/8*f


def OmegaGW_qmc(k, num_samples=300000):
    sampler = qmc.Sobol(d=2, scramble=True, seed = 42)
    raw_samples = sampler.random(num_samples)
    
    l_bounds = [-1, 0]
    u_bounds = [1, 50]
    samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    samples = jnp.array(samples)
    
    intvals = vmap(lambda args: IuvPB238_qmc(args, k))(samples)
    volume = jnp.prod(jnp.array(u_bounds) - jnp.array(l_bounds))
    integral = jnp.mean(intvals) * volume
    # t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = Ci(-k*TR)**2+(pi/2+Si(-k*TR))**2
    # jax.debug.print("t2:{}", t2)

    return (t2*integral)
Pi0 = 1e6
k_vals = jnp.logspace(-2,3,1000)
# Omeg = vmap(OmegaGW_qmc)(k_vals)
Omeg = jnp.array(list(map(OmegaGW_qmc, k_vals)))

#%%
plt.loglog(k_vals, -Omeg)
# plt.loglog(k_vals, np.abs(Omeg))

# plt.loglog(k_vals, 2e3*prayer,'--', label = "pow3*2e3")
# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
# plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"QMC Sobol: kstar = 1000, $\Pi_0$ = 1e6, $\tau_R$=-0.01")
# plt.ylim(top = 10**5)
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.png', bbox_inches='tight')

plt.show()
#%%
Pi0 = 10
Omeg2 = jnp.array(list(map(OmegaGW_qmc, k_vals)))
#%%

plt.loglog(k_vals, -Omeg, label=r"$\Pi_0$=1e6")
plt.loglog(k_vals, Omeg2, label = r"$\Pi_0$=10")


# plt.loglog(k_vals, 2e3*prayer,'--', label = "pow3*2e3")
# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Sobol QMC: kstar = 1000, $\tau_R$=-0.01")
# plt.ylim(top = 10**5)
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW_Sobol_C0v2_Pi0_comp2.png', bbox_inches='tight')

plt.show()