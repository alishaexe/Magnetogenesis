import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
def f(kappa, Pi0):
    term1 = 1 + (1 / kappa**6) * 4 * (1 + kappa**2) * Pi0**2 * (3 * kappa * np.cos(kappa) + (-3 + kappa**2) * np.sin(kappa))**2
    term2 = (1 / kappa**3) * 2 * Pi0 * (kappa**3 - kappa * (-6 + kappa**2) * np.cos(2 * kappa) + (-3 + 4 * kappa**2) * np.sin(2 * kappa))
    return term1 + term2

def power_law(kappa, A, B):
    return A * kappa**B

# Define range of kappa values
kappa_vals = np.linspace(0.01, 1e3, 10000000)  
Pi0 = 1000  

# Compute function values
f_vals = f(kappa_vals, Pi0)

# Plot the function
plt.loglog(kappa_vals, f_vals, label=r'$\Pi(\kappa)$')
plt.xlabel(r'$\kappa$')
plt.ylabel(r'$\Pi(\kappa)$')
# plt.ylim(1e-6,5e7)

# plt.title('Plot of the function')
plt.legend()
plt.grid()
plt.show()

#%%
# # Compute numerical gradient
# gradient_vals = np.gradient(f_vals, kappa_vals)

# # Define the range where you want to analyze the gradient
# kappa_range = (kappa_vals > 0.2) & (kappa_vals < 2)  # Example: choosing kappa in [2, 5]

# # Plot the gradient for the selected range
# plt.loglog(kappa_vals[kappa_range], gradient_vals[kappa_range], label=r'Gradient of $f(\kappa)$', color='red')
# plt.xlabel(r'$\kappa$')
# plt.ylabel(r'$\frac{df}{d\kappa}$')
# plt.title('Gradient of the function')
# plt.legend()
# plt.grid()
# plt.show()

#%%
# Select the range where you want to fit the power law
fit_range = (kappa_vals > 0.2) & (kappa_vals < 2)

kappa_fit = kappa_vals[fit_range]
f_fit = f_vals[fit_range]

# Perform power-law fitting
popt, pcov = curve_fit(power_law, kappa_fit, f_fit)

# Extract fitted parameters
A_fit, B_fit = popt
print(f"Fitted Power-Law: A = {A_fit:.4f}, B = {B_fit:.4f}")

# Generate fitted curve
fitted_curve = power_law(kappa_fit, A_fit, B_fit)

# Plot the original function
plt.loglog(kappa_vals, f_vals, label=r'Original $f(\kappa)$', color='blue', alpha=0.5)
# Plot the selected data for fitting
plt.scatter(kappa_fit, f_fit, color='black', label='Data for Fit')
# Plot the power-law fit
plt.loglog(kappa_fit, fitted_curve, label=r'Power-Law Fit: $A \kappa^B$', color='red')

plt.xlabel(r'$\kappa$')
plt.ylabel(r'$f(\kappa)$')
plt.title('Power-Law Fit to Selected Part of the Curve')
plt.legend()
plt.grid()
plt.show()

#%%
kappa_range = (kappa_vals > 0.1) & (kappa_vals < 3)  # Example: choosing kappa in [2, 5]

grad = power_law(kappa_vals[kappa_range], A_fit, B_fit)
# Plot the function
plt.loglog(kappa_vals, f_vals, 'black',label=r'$\Pi(\kappa)$')
plt.loglog(kappa_vals[kappa_range], grad,'--r' ,label = r'Power-Law Fit: $A \kappa^B$' )
plt.xlabel(r'$\kappa$')
plt.ylabel(r'$\Pi(\kappa)$')
plt.legend()
plt.grid()
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
# plt.ylim(1e-4,5e7)
plt.savefig('/Users/alisha/Documents/Magnetogenesis/Plots/fittedPL.png', bbox_inches='tight')

plt.show()
