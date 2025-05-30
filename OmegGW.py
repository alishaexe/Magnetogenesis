import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import sici, exp1
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

def C0(t,s):
    t1num = (s**2+t*(t+2)-1)**2
    t1denom = 4*(-s+t+1)**2
    
    t2num = (s**2+t*(t+2)+3)**2
    t2denom = 4*(s+t+1)**2
    
    t1 = (t1num/t1denom)+1
    t2 = (t2num/t2denom)+1

    return t1*t2

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




def PB238(k):
    if k>50:
        return 0
    else:
        res = 1 +2/k**3 * Pi0*(k**3+(4*k**2-3)*np.sin(2*k)-(k**2-6)*k*np.cos(2*k))+4/k**6*Pi0**2*(k**2+1)*((k**2-3)*np.sin(k)+3*k*np.cos(k))**2
        return res#/(1+0.01*k**8)                                                                        

def IuvPB238(k):
    f = lambda s, t: 1/(1-s+t)**2*1/(1+s+t)**2*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)
    # res = dblquad(f,0,np.inf,-1,1)[0]
    res = dblquad(f, 0, np.inf, -1, 1,epsabs=1e-5, epsrel=1e-4)[0]
    return res



def OmegGWpb238(k):
    
    # t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = (Ci(k*xstar)**2+(pi/2+Si(k*xstar))**2)
    return t2*IuvPB238(k)


start = time.time()
xstar = 1e-3
Pi0 = 1e6
k_vals = np.logspace(-5,2,1000)
prayer = np.abs(np.array(list(map(OmegGWpb238, k_vals))))

end = time.time()
print("Calculation Time:", end-start)
#%%
# np.save("/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.np",prayer)
# RELOAD THE FILE!!!!
plt.loglog(k_vals, prayer ,label = "")
# plt.loglog(k_vals, np.abs(prayer) ,label = "abs PB238")
# plt.loglog(k_vals, bill_pb ,label = "Bill Pb")

# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
# plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Scipy numerical integration")
# plt.ylim(top = 10**5)
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW.png', bbox_inches='tight')

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