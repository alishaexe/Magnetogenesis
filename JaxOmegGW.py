import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import sici, exp1
import jax
import jax.numpy as jnp
from jax import jit, vmap

#%%
HI = 1e-6
TR=-0.01
pi = np.pi
alpha = 1e7
Delta_tau = 1e-1
# Pi0 = 1e6
kstar = 1000
#%%
# def C0(t,s):
#     t1 = s**4*(t*(t+2)+2)**2
#     t2 = 4*s**3*t*(t+1)*(t+2)
#     t3 = 2*s**2*(t-2)*(t*(t+4)+2)
#     tdenom = (s**2-(t+1)**2)**2
#     if abs(tdenom)<1e-4:
#         return 0
    
#     t4 = -(4*s*t*(t+1)*(t+2)+(t*(t+2)+2)**2)
#     return (t1+t2+t3)/tdenom+t4/tdenom
@jit
def C0(t,s):
    t1num = (s**2+t*(t+2)+3)**2*(s**2+t*(t+2)-1)**2
    t1denom = 16*(-s+t+1)**2*(s+t+1)**2
    if t1denom<1e-4:
        t1 = 0
    else:
        t1 = t1num/t1denom
    t2num = (s**2+t*(t+2)-1)**2
    t2denom = 2*(-s+t+1)**2
    if t2denom<1e-4:
        t2 = 0
    else:
        t2 = t2num/t2denom
    
    return 1 + t1 + t2


def Si(x):
    # integrand = lambda xb: np.sin(xb)/xb
    # res = quad(integrand, 0, x)[0]
    res = sici(x)[0]
    return res

def Ci(x):
    # integrand = lambda xb: np.cos(xb)/xb
    # res = -quad(integrand, x, np.inf)[0]
    res = sici(x)[1]
    
    return res



@jit
def PB238(kappa):
    
    
    # if kappa <= 1e3:
    #     f = 1
    # if kappa >1e3:
    #     f=0
    # a = 1000 #Smoothness parameter
    cutoff = 1 #/ (1 + np.exp(a * (kappa - 1000)))
    if kappa>1/xstar:
        return 0
    term1 = (1+ 2*Pi0*(kappa**3+(4*kappa**2-3)*jnp.sin(2*kappa)-(kappa*2-6)*kappa*jnp.cos(2*kappa))/kappa**3)*cutoff 
    term2 = ((4*(kappa**2+1)*Pi0**2*((kappa**2-3)*jnp.sin(kappa)+3*kappa*jnp.cos(kappa))**2)/kappa**6)*cutoff
    return (term1+term2)#*(9*HI**4)/(4*pi**2) #Pi0 = Pb/pb(k<<1)
                                            #So Pb = Pi0*Pb(k<<1)
                           # No longer using Pb -> using Pi instead                                                                         

def IuvPB238(k):
    f = lambda s, t: 1/(1-s+t)**2*1/(1+s+t)**2*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)
    # res = dblquad(f,0,jnp.inf,-1,1)[0]
    res = dblquad(f, 0, jnp.inf, -1, 1,epsabs=1e-5, epsrel=1e-4)[0]
    return 1/8*res



def OmegGWpb238(k):
    
    # t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = k**2*Ci(k*xstar)**2+(pi/2+Si(k*xstar))**2
    return t2*IuvPB238(k)


start = time.time()
xstar = 1e-3
Pi0 = 1e6
k_vals = jnp.logspace(-2,6,1000)/kstar
# prayer = jnp.abs(jnp.array(list(map(OmegGWpb238, k_vals))))
prayer = jnp.abs(vmap(OmegGWpb238)(k_vals))
end = time.time()
print("Calculation Time:", end-start)
#%%
# np.save("/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.np",prayer)
# RELOAD THE FILE!!!!
plt.loglog(k_vals, prayer ,label = "PB238")
# plt.loglog(k_vals, np.abs(prayer) ,label = "abs PB238")
# plt.loglog(k_vals, bill_pb ,label = "Bill Pb")

# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Python: kstar = 1000, $\Pi_0$ = 1e6, $x_\star$=1e-3")
# plt.ylim(top = 10**5)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.png', bbox_inches='tight')

plt.show()

#%%

Pi0 = 10
hope = np.abs(np.array(list(map(OmegGWpb238, k_vals))))


#%%
plt.loglog(k_vals, hope, label="PB238")
# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"kstar = 1000, $\Pi_0$ = 10, $\tau_R$=-0.01")
# plt.ylim(top = 10**5)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW_C0v2_Pi0_10.png', bbox_inches='tight')

plt.show()

#%%
plt.loglog(k_vals, prayer ,label = r"$\Pi_0$=1e6")
plt.loglog(k_vals, hope, label=r"$\Pi_0$=10")
plt.xlabel(r"$\kappa$", size = 14)
plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Python numerics: kstar = 1000, $\tau_R$=-0.01")
# plt.ylim(top = 10**5)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW_C0v2_Pi0_comp.png', bbox_inches='tight')

plt.show()