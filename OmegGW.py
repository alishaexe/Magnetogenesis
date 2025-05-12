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

# def C0(t, s):
#     u = (t+s+1)/2
#     v = (t-s+1)/2
#     # mu = (1-u**2+v**2)/(2*v)
#     term1 = -u**4/2+v**6/(8*u**2)+1/2*(1/u**2-1)*v**4
#     term2 = 1/4*(3*u**2-7/u**2-2)*v**2-u**2/2 +  1/u**2
#     term3 = ((u**2-1)**2*(u**4+6*u**2+1))/(8*u**2*v**2) + 5
#     res = term1+term2+term3
#     return res

# def C0(t,s):
#     term1 = s**4*(t*(t+2)+2)**2
#     term2 = 2*s**2*(t-2)*(t*(t+4)+2)
#     term3 = (t*(t+2)+2)**2
#     term4 = (s**2-(t+1)**2)**2
#     return 2*(term1+term2+term3)/term4

def C0(t,s):
    t1 = s**4*(t*(t+2)+2)**2
    t2 = 4*s**3*t*(t+1)*(t+2)
    t3 = 2*s**2*(t-2)*(t*(t+4)+2)
    tdenom = (s**2-(t+1)**2)**2
    if abs(tdenom)<1e-4:
        return 0
    
    t4 = -(4*s*t*(t+1)*(t+2)+(t*(t+2)+2)**2)
    return (t1+t2+t3)/tdenom+t4/tdenom

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


# def PB(k):
#     # A = 32236.4207
#     A = 100
#     B = 4.6226
#     return A*k**B

def PB(k):
    kappa = k/kstar
    denominator = 64 * (-1 + Delta_tau)**8 * kappa**6
    
    term1 = np.exp(2j * kappa) * alpha * (-3j + 2 * (-1 + Delta_tau) * kappa)
    term2 = np.exp(4j * Delta_tau * kappa) * alpha * (-3j * (1 + 4 * Delta_tau) + 
            2 * (2 + (7 - 3 * Delta_tau) * Delta_tau) * kappa -
            2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa**2)
    term3 = np.exp(2j * Delta_tau * kappa) * alpha * (3j + 2 * (-1 + Delta_tau) * kappa * 
            (2 - 1j * (-1 + Delta_tau) * kappa))
    term4 = np.exp(2j * (1 + Delta_tau) * kappa) * (
            8 * (-1 + Delta_tau)**4 * kappa**3 + alpha * (3j + 2 * kappa +
            2 * Delta_tau * (6j + kappa * (2 - 9 * Delta_tau - 
            2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa + 
            2 * (-1 + Delta_tau)**2 * (1 + Delta_tau) * kappa**2))))
    
    numerator1 = (np.exp(1j * (-2 + 3 * Delta_tau) * kappa) * alpha * (3j + 2 * (-1 + Delta_tau) * kappa) +
                  (alpha * (3j * (1 + 4 * Delta_tau) + 2 * (2 + (7 - 3 * Delta_tau) * Delta_tau) * kappa +
                  2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa**2)) / np.exp(1j * Delta_tau * kappa) +
                  np.exp(1j * Delta_tau * kappa) * alpha * (-3j + 2 * (-1 + Delta_tau) * kappa *
                  (2 + 1j * (-1 + Delta_tau) * kappa)) +
                  np.exp(1j * (-2 + Delta_tau) * kappa) * (
                      8 * (-1 + Delta_tau)**4 * kappa**3 + alpha * (-3j + 2 * kappa +
                      2 * Delta_tau * (-6j + kappa * (2 - 9 * Delta_tau +
                      2j * (-1 + Delta_tau) * (1 + 3 * Delta_tau) * kappa +
                      2 * (-1 + Delta_tau)**2 * (1 + Delta_tau) * kappa**2)))))
    
    numerator = (term1 + term2 + term3 + term4) * numerator1 / np.exp(3j * Delta_tau * kappa)
    return np.real(numerator / denominator * (9*HI**4)/(4*pi**2))/kappa**3

# def PBpow3(kappa):
#     Pi0 = 1000
#     term1 = 1 + (1 / kappa**6) * 4 * (1 + kappa**2) * Pi0**2 * (3 * kappa * np.cos(kappa) + (-3 + kappa**2) * np.sin(kappa))**2
#     term2 = (1 / kappa**3) * 2 * Pi0 * (kappa**3 - kappa * (-6 + kappa**2) * np.cos(2 * kappa) + (-3 + 4 * kappa**2) * np.sin(2 * kappa))
#     return (term1 + term2)* (9*HI**4)/(4*pi**2)

def PB238(k):
    kappa = k/kstar
    
    # if kappa <= 1e3:
    #     f = 1
    # if kappa >1e3:
    #     f=0
    a = 1000 #Smoothness parameter
    cutoff = 1 / (1 + np.exp(a * (kappa - 1000)))
    
    term1 = (1+ 2*Pi0*(kappa**3+(4*kappa**2-3)*np.sin(2*kappa)-(kappa*2-6)*kappa*np.cos(2*kappa))/kappa**3)*cutoff 
    term2 = ((4*(kappa**2+1)*Pi0**2*((kappa**2-3)*np.sin(kappa)+3*kappa*np.cos(kappa))**2)/kappa**6)*cutoff
    return (term1+term2)*(9*HI**4)/(4*pi**2) #Pi0 = Pb/pb(k<<1)
                                            #So Pb = Pi0*Pb(k<<1)
                                                                                                    
def Iuv(k):
    f = lambda s, t: (1-s+t)*(1+s+t)*PB(k*((t+s+1)/2))*PB(k*((t-s+1)/2))*C0(t,s)
    res = dblquad(f,0,np.inf,-1,1)[0]
    return 1/8*res

# def Iuvpow3(k):
#     f = lambda s, t: (1-s+t)*(1+s+t)*PBpow3(k*((t+s+1)/2))*PBpow3(k*((t-s+1)/2))*C0(t,s)
#     res = dblquad(f,0,np.inf,-1,1)[0]
#     return 1/8*res

def IuvPB238(k):
    f = lambda s, t: (1-s+t)*(1+s+t)*PB238(k*((t+s+1)/2))*PB238(k*((t-s+1)/2))*C0(t,s)
    # res = dblquad(f,0,np.inf,-1,1)[0]
    res = dblquad(f, 0, 250, lambda t: -1, lambda t: 1,epsabs=1e-5, epsrel=1e-4)[0]
    return 1/8*res


def OmegGW(k):
    # t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = Ci(-k*TR)**2+(pi/2+Si(-k*TR))**2
    return t2*Iuv(k)

def OmegGWpb238(k):
    # t1 = (HI**4*(-k*TR)**8)/(24576*pi**6)
    t2 = Ci(-k*TR)**2+(pi/2+Si(-k*TR))**2
    return t2*IuvPB238(k)
start = time.time()
Pi0 = 1e6
k_vals = np.logspace(-2,6,1000)
# hope = np.abs(np.array(list(map(OmegGW, k_vals))))
prayer = np.abs(np.array(list(map(OmegGWpb238, k_vals))))

# bill_pb = np.abs(np.array(list(map(OmegGW, k_vals))))
end = time.time()
print("Calculation Time:", end-start)
#%%
plt.loglog(k_vals, prayer ,label = "PB238")
# plt.loglog(k_vals, np.abs(prayer) ,label = "abs PB238")
# plt.loglog(k_vals, bill_pb ,label = "Bill Pb")

# plt.xlim(1e-3,1e-1)
plt.xlabel(r"$\kappa$", size = 14)
plt.legend()
plt.ylabel(r"$\Omega_{GW}$", size = 16)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.title(r"Python: kstar = 1000, $\Pi_0$ = 1e6, $\tau_R$=-0.01")
# plt.ylim(top = 10**5)
# plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/OmegGW_C0v2_Pi0_1e6.png', bbox_inches='tight')

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